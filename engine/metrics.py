"""
Presence Tracker — Metric Computation Engine

Computes the 5 derived metrics for each 15-minute observation window:
- Cognitive Load Score (CLS)
- Focus Depth Index (FDI)
- Social Drain Index (SDI)
- Context Switch Cost (CSC)
- Recovery Alignment Score (RAS)

All metrics are normalized to [0.0, 1.0].

v1.1 — HRV-aware physiological composite:
  HRV (rmssd) and sleep_performance are now incorporated into the
  physiological readiness signal used by CLS and RAS.  Previously only
  recovery_score was considered; low HRV (autonomic stress marker) and
  poor sleep now correctly raise the perceived cognitive load baseline
  and reduce the physiological capacity available for work.

v1.2 — RescueTime integration:
  When RescueTime data is available, CLS, FDI, and CSC are upgraded from
  proxy-based to signal-based computation:

  CLS: productivity_score replaces the recovery-inverse proxy for
       application-level cognitive demand.  A distracted window (low
       productivity_score) raises CLS even when WHOOP recovery is high.
       Weight is blended: 0.25 RT signal + 0.75 existing formula so that
       the metric degrades gracefully when RT is absent.

  FDI: app_switches replaces the Slack-channels proxy for context switching.
       Real app-switch counts from RescueTime are a direct measure of
       fragmentation — far more precise than cross-channel Slack activity.

  CSC: app_switches from RescueTime are added as a third component
       alongside meeting switches and Slack channel switches, providing
       a real behavioral signal for fragmentation cost.

  All three functions accept optional rescuetime_data kwargs so existing
  callers without RT data continue to work unchanged.
"""

from typing import Optional

# ─── Constants ───────────────────────────────────────────────────────────────

# Maximum RescueTime app switches per 15-min window before the signal saturates.
# 8 switches = highly fragmented (switching apps roughly every 2 minutes).
_MAX_RT_APP_SWITCHES = 8


# ─── Normalization helpers ────────────────────────────────────────────────────

def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _norm(val: Optional[float], max_val: float, min_val: float = 0.0) -> float:
    """Normalize a value to [0, 1] given expected range."""
    if val is None:
        return 0.0
    if max_val == min_val:
        return 0.0
    return _clamp((val - min_val) / (max_val - min_val))


# ─── Physiological readiness composite ───────────────────────────────────────

# Reference HRV: population median for a healthy adult.
# Values below this indicate autonomic stress; above indicate readiness.
_HRV_REFERENCE_MS = 65.0

# HRV saturation point: at this value (or above) HRV contributes its maximum
# positive signal.  Prevents extreme outliers from dominating.
_HRV_SATURATION_MS = 130.0

# Weights for the composite readiness score
_READINESS_W_RECOVERY = 0.50   # WHOOP recovery score (already composite)
_READINESS_W_HRV = 0.30        # HRV RMSSD — real-time autonomic state
_READINESS_W_SLEEP = 0.20      # Sleep performance — substrate for the day


def physiological_readiness(
    recovery_score: Optional[float],
    hrv_rmssd_milli: Optional[float] = None,
    sleep_performance: Optional[float] = None,
) -> float:
    """
    Composite physiological readiness score — 0.0 (depleted) to 1.0 (peak).

    Blends three WHOOP signals:
    - recovery_score (0–100): WHOOP's own composite readiness
    - hrv_rmssd_milli: heart-rate variability; low HRV = autonomic stress
    - sleep_performance (0–100): how well the previous sleep prepared the body

    When individual signals are missing the weight is redistributed to
    available signals.  If all signals are None, returns 0.5 (neutral).

    This composite is used as the physiological capacity signal in CLS and RAS.
    """
    components: list[tuple[float, float]] = []  # (value_0_to_1, weight)

    if recovery_score is not None:
        components.append((recovery_score / 100.0, _READINESS_W_RECOVERY))

    if hrv_rmssd_milli is not None:
        # Normalize HRV: 0 at 0ms, 0.5 at reference, 1.0 at saturation.
        # Uses a two-stage linear scale so the reference is exactly the midpoint.
        if hrv_rmssd_milli <= _HRV_REFERENCE_MS:
            hrv_norm = 0.5 * (hrv_rmssd_milli / _HRV_REFERENCE_MS)
        else:
            hrv_norm = 0.5 + 0.5 * min(
                1.0,
                (hrv_rmssd_milli - _HRV_REFERENCE_MS) / (_HRV_SATURATION_MS - _HRV_REFERENCE_MS)
            )
        components.append((hrv_norm, _READINESS_W_HRV))

    if sleep_performance is not None:
        components.append((sleep_performance / 100.0, _READINESS_W_SLEEP))

    if not components:
        return 0.5  # All signals missing → neutral

    # Redistribute weights so they always sum to 1.0
    total_weight = sum(w for _, w in components)
    readiness = sum(v * (w / total_weight) for v, w in components)

    return round(_clamp(readiness), 4)


# ─── CLS: Cognitive Load Score ───────────────────────────────────────────────

def cognitive_load_score(
    in_meeting: bool,
    meeting_attendees: int,
    slack_messages_received: int,
    recovery_score: Optional[float],
    hrv_rmssd_milli: Optional[float] = None,
    sleep_performance: Optional[float] = None,
    window_duration_minutes: int = 15,
    rt_productivity_score: Optional[float] = None,
    rt_active_seconds: int = 0,
) -> float:
    """
    Cognitive Load Score — how mentally demanding was this window?

    Range: 0.0 (completely idle/recovered) to 1.0 (maximum load)

    Inputs:
        in_meeting: whether a calendar event was active
        meeting_attendees: number of participants in active meeting
        slack_messages_received: incoming Slack messages in this window
        recovery_score: WHOOP recovery (0-100), None if unavailable
        hrv_rmssd_milli: HRV in ms (v1.1 — improves physiological baseline)
        sleep_performance: WHOOP sleep performance (0-100), optional
        rt_productivity_score: RescueTime productivity score (0.0–1.0).
            1.0 = very productive, 0.0 = very distracting.  When provided,
            used as a direct behavioral signal for cognitive demand.
            Low productivity_score → high distraction → higher CLS.
            (v1.2 — replaces recovery_inverse as the behavioral demand proxy)
        rt_active_seconds: total active computer time in this window (0–900).
            Used to gate the RT signal: if the computer was idle, RT
            distraction data is not meaningful.
    """
    # Component: meeting density
    # 1.0 if in a meeting, 0.0 if not
    meeting_component = 1.0 if in_meeting else 0.0

    # Component: calendar pressure
    # Normalized by max expected attendees (10)
    calendar_pressure = _norm(meeting_attendees if in_meeting else 0, max_val=10)

    # Component: Slack volume
    # 30 messages in 15 min = saturation
    slack_component = _norm(slack_messages_received, max_val=30)

    # Component: Recovery inverse (low readiness = higher cognitive load baseline)
    # v1.1: uses composite physiological readiness (recovery + HRV + sleep)
    # instead of recovery_score alone.  Low HRV or poor sleep raises the baseline.
    readiness = physiological_readiness(recovery_score, hrv_rmssd_milli, sleep_performance)
    recovery_inverse = 1.0 - readiness

    # v1.2: RescueTime behavioral demand signal
    # When the computer was actively used and RT data is available, blend in
    # the distraction signal (1 - productivity_score) as a real behavioral
    # measure of cognitive demand.
    #
    # Blend formula: CLS = 0.75 * base_cls + 0.25 * rt_demand
    # This keeps the existing formula dominant while RT adds a meaningful nudge.
    # rt_demand is only applied when the machine was active (>= 60s) to avoid
    # zero-productivity scores for truly idle windows pulling CLS up.

    base_cls = (
        0.35 * meeting_component +
        0.20 * calendar_pressure +
        0.25 * slack_component +
        0.20 * recovery_inverse
    )

    if rt_productivity_score is not None and rt_active_seconds >= 60:
        # Low productivity score = high distraction = higher demand
        rt_demand = 1.0 - rt_productivity_score
        cls = 0.75 * base_cls + 0.25 * rt_demand
    else:
        cls = base_cls

    return round(_clamp(cls), 4)


# ─── FDI: Focus Depth Index ──────────────────────────────────────────────────

def focus_depth_index(
    in_meeting: bool,
    slack_messages_received: int,
    context_switches: int = 0,
    rt_app_switches: Optional[int] = None,
    rt_active_seconds: int = 0,
    rt_productivity_score: Optional[float] = None,
) -> float:
    """
    Focus Depth Index — how deep was the focus in this window?

    Range: 0.0 (completely fragmented) to 1.0 (deep, uninterrupted focus)

    Inputs:
        in_meeting: whether a calendar event was active
        slack_messages_received: incoming messages
        context_switches: app switches proxy (Slack channels), used when
            RescueTime data is not available
        rt_app_switches: real app switch count from RescueTime (v1.2).
            When provided and the machine was active, replaces the Slack
            channel proxy with a direct behavioral measure.  Saturates
            at 8 switches per 15 minutes (highly fragmented).
        rt_active_seconds: active computer time; gates the RT signal.
        rt_productivity_score: RescueTime productivity (0–1).  When
            available, incorporated as an additional focus signal: a
            distracted window has lower FDI independent of app switches.

    v1.2: When RescueTime data is present (rt_app_switches is not None
    and rt_active_seconds >= 60), real app-switch counts replace the
    Slack-channels proxy for context switching, improving precision of
    the FDI signal significantly.
    """
    # Meetings break focus
    meeting_disruption = 1.0 if in_meeting else 0.0

    # Slack interruptions
    slack_disruption = _norm(slack_messages_received, max_val=30)

    # Context switches:
    # v1.2: use real RT app_switches when available (saturates at 8 per window)
    # Fallback: Slack channels proxy (saturates at 20 for backward compat)
    if rt_app_switches is not None and rt_active_seconds >= 60:
        switch_disruption = _norm(rt_app_switches, max_val=_MAX_RT_APP_SWITCHES)
    else:
        switch_disruption = _norm(context_switches, max_val=20)

    # v1.2: RT distraction signal — if the window was active but unproductive,
    # that's a direct focus-fragmentation signal even without explicit app switches.
    # Only applied when the machine was active so idle windows aren't penalised.
    if rt_productivity_score is not None and rt_active_seconds >= 60:
        # High distraction (low productivity) = additional disruption signal
        rt_distraction = 1.0 - rt_productivity_score
        # Weight: 0.80 existing disruption + 0.20 RT distraction blend
        disruption = 0.80 * (
            0.40 * meeting_disruption +
            0.40 * slack_disruption +
            0.20 * switch_disruption
        ) + 0.20 * rt_distraction
    else:
        disruption = (
            0.40 * meeting_disruption +
            0.40 * slack_disruption +
            0.20 * switch_disruption
        )

    # FDI = 1 - disruption score
    return round(_clamp(1.0 - disruption), 4)


# ─── SDI: Social Drain Index ─────────────────────────────────────────────────

def social_drain_index(
    in_meeting: bool,
    meeting_attendees: int,
    slack_messages_sent: int,
    slack_messages_received: int,
) -> float:
    """
    Social Drain Index — how much social energy was expended?

    Range: 0.0 (isolated/quiet) to 1.0 (maximum social engagement)

    Higher values indicate more social interactions that may drain energy.
    """
    # Large meetings drain more energy
    attendee_component = _norm(meeting_attendees if in_meeting else 0, max_val=10)

    # Any meeting at all
    meeting_component = 1.0 if in_meeting else 0.0

    # Sent messages (active communication) vs received (passive)
    total_slack = slack_messages_sent + slack_messages_received
    if total_slack > 0:
        sent_ratio = slack_messages_sent / total_slack
    else:
        sent_ratio = 0.0

    sdi = (
        0.50 * attendee_component +
        0.30 * meeting_component +
        0.20 * sent_ratio
    )

    return round(_clamp(sdi), 4)


# ─── CSC: Context Switch Cost ────────────────────────────────────────────────

def context_switch_cost(
    in_meeting: bool,
    meeting_duration_minutes: int,
    slack_channels_active: int,
    is_short_meeting: bool = False,
    rt_app_switches: Optional[int] = None,
    rt_active_seconds: int = 0,
) -> float:
    """
    Context Switch Cost — fragmentation penalty for this window.

    Range: 0.0 (no switching, sustained mode) to 1.0 (maximum fragmentation)

    Short meetings (<30 min) are costlier than long ones because they force
    rapid context switches without settling into deep work.

    v1.2: When RescueTime data is available, real app-switch counts are
    incorporated as an additional behavioral component.  This captures
    digital context switching (IDE → browser → Slack → email) that the
    existing meeting/channel signals cannot detect.

    With RT: weights shift to 0.40 meetings, 0.25 channels, 0.20 calendar
             fragmentation, 0.15 RT app switches.
    Without RT: original weights (0.50/0.30/0.20).
    """
    # Short meetings are more costly context switches
    if in_meeting:
        meeting_switch = 0.5 + (0.5 if is_short_meeting else 0.0)
    else:
        meeting_switch = 0.0

    # Cross-channel Slack activity
    channel_switch = _norm(slack_channels_active, max_val=5)

    # Calendar fragmentation: being in a very short meeting (<15 min)
    fragmentation = 1.0 if (in_meeting and meeting_duration_minutes < 15) else (
        0.5 if (in_meeting and meeting_duration_minutes < 30) else 0.0
    )

    # v1.2: real app-switch signal from RescueTime
    if rt_app_switches is not None and rt_active_seconds >= 60:
        app_switch_component = _norm(rt_app_switches, max_val=_MAX_RT_APP_SWITCHES)
        csc = (
            0.40 * meeting_switch +
            0.25 * channel_switch +
            0.20 * fragmentation +
            0.15 * app_switch_component
        )
    else:
        csc = (
            0.50 * meeting_switch +
            0.30 * channel_switch +
            0.20 * fragmentation
        )

    return round(_clamp(csc), 4)


# ─── RAS: Recovery Alignment Score ───────────────────────────────────────────

def recovery_alignment_score(
    recovery_score: Optional[float],
    cls: float,
    hrv_rmssd_milli: Optional[float] = None,
    sleep_performance: Optional[float] = None,
) -> float:
    """
    Recovery Alignment Score — is your activity level appropriate for your physiology?

    Range: 0.0 (badly misaligned) to 1.0 (perfectly aligned)

    High readiness + low CLS = aligned (resting on a rest day) → high RAS
    High readiness + high CLS = aligned (working hard when ready) → high RAS
    Low readiness + low CLS = semi-aligned (resting when tired) → medium RAS
    Low readiness + high CLS = misaligned (pushing hard when depleted) → low RAS

    v1.1: capacity_available uses the composite physiological readiness
    (recovery + HRV + sleep) rather than recovery_score alone.
    Low HRV or poor sleep correctly reduce the available capacity, making
    the misalignment signal more sensitive and accurate.

    When all physiological signals are unavailable, returns 0.5 (neutral)
    to avoid producing a meaningless alignment score.
    """
    # If all physiological signals are absent, we have no basis to compute
    # alignment — return neutral rather than running the capacity model on
    # a manufactured 0.5 readiness.
    if recovery_score is None and hrv_rmssd_milli is None and sleep_performance is None:
        return 0.5

    # v1.2: full composite readiness as capacity measure
    capacity_available = physiological_readiness(
        recovery_score, hrv_rmssd_milli, sleep_performance
    )

    capacity_used = cls

    # Alignment is modelled as a smooth, monotonic function of (capacity - demand).
    # The signed margin tells us how comfortable the load is relative to capacity:
    #   positive margin → working within capacity → good alignment
    #   negative margin → overloading capacity    → poor alignment
    #
    # Formula: RAS = 0.5 + 0.5 * tanh(k * margin)
    #   k controls sensitivity; k=3 gives a smooth S-curve where a 0.33-unit
    #   margin (e.g. capacity 0.83, CLS 0.50) yields RAS ≈ 0.90 and a -0.33
    #   margin yields RAS ≈ 0.10.
    #
    # Properties:
    #   - Strictly monotone: higher capacity always improves RAS for the same CLS
    #   - Continuous and differentiable (no discontinuous branch transition)
    #   - Bounded [0, 1] by construction
    #   - Neutral (0.5) when capacity exactly equals demand
    import math
    _K = 3.0
    margin = capacity_available - capacity_used
    ras = 0.5 + 0.5 * math.tanh(_K * margin)

    return round(_clamp(ras), 4)


# ─── Master metric computation ───────────────────────────────────────────────

def compute_metrics(window_data: dict) -> dict:
    """
    Compute all 5 metrics for a single window.

    window_data must contain:
    - calendar: {in_meeting, meeting_attendees, meeting_duration_minutes}
    - whoop: {recovery_score, hrv_rmssd_milli, sleep_performance}
    - slack: {messages_sent, messages_received, channels_active}

    Optional (v1.2):
    - rescuetime: {app_switches, productivity_score, active_seconds, ...}
      When present and the machine was active, RescueTime signals upgrade
      FDI, CSC, and CLS from proxy-based to signal-based computation.

    v1.1: hrv_rmssd_milli and sleep_performance are now forwarded to CLS
    and RAS so the physiological readiness composite is fully utilised.

    v1.2: rescuetime sub-dict is now extracted and forwarded to CLS, FDI,
    and CSC so that real behavioral signals (app_switches, productivity_score)
    are used when available, replacing the Slack-channel proxies.
    """
    cal = window_data.get("calendar", {})
    whoop = window_data.get("whoop", {})
    slack = window_data.get("slack", {})
    rt = window_data.get("rescuetime") or {}  # Optional — None-safe

    in_meeting = cal.get("in_meeting", False)
    meeting_attendees = cal.get("meeting_attendees", 0)
    meeting_duration = cal.get("meeting_duration_minutes", 0)
    recovery_score = whoop.get("recovery_score")
    hrv_rmssd_milli = whoop.get("hrv_rmssd_milli")
    sleep_performance = whoop.get("sleep_performance")
    slack_sent = slack.get("messages_sent", 0)
    slack_received = slack.get("messages_received", 0)
    slack_channels = slack.get("channels_active", 0)

    # RescueTime signals (None when RT is not configured/available)
    rt_app_switches: Optional[int] = rt.get("app_switches") if rt else None
    rt_productivity_score: Optional[float] = rt.get("productivity_score") if rt else None
    rt_active_seconds: int = int(rt.get("active_seconds", 0)) if rt else 0

    cls = cognitive_load_score(
        in_meeting=in_meeting,
        meeting_attendees=meeting_attendees,
        slack_messages_received=slack_received,
        recovery_score=recovery_score,
        hrv_rmssd_milli=hrv_rmssd_milli,
        sleep_performance=sleep_performance,
        rt_productivity_score=rt_productivity_score,
        rt_active_seconds=rt_active_seconds,
    )

    fdi = focus_depth_index(
        in_meeting=in_meeting,
        slack_messages_received=slack_received,
        context_switches=slack_channels,  # Proxy when RescueTime unavailable
        rt_app_switches=rt_app_switches,
        rt_active_seconds=rt_active_seconds,
        rt_productivity_score=rt_productivity_score,
    )

    sdi = social_drain_index(
        in_meeting=in_meeting,
        meeting_attendees=meeting_attendees,
        slack_messages_sent=slack_sent,
        slack_messages_received=slack_received,
    )

    csc = context_switch_cost(
        in_meeting=in_meeting,
        meeting_duration_minutes=meeting_duration,
        slack_channels_active=slack_channels,
        is_short_meeting=(meeting_duration < 30) if in_meeting else False,
        rt_app_switches=rt_app_switches,
        rt_active_seconds=rt_active_seconds,
    )

    ras = recovery_alignment_score(
        recovery_score=recovery_score,
        cls=cls,
        hrv_rmssd_milli=hrv_rmssd_milli,
        sleep_performance=sleep_performance,
    )

    return {
        "cognitive_load_score": cls,
        "focus_depth_index": fdi,
        "social_drain_index": sdi,
        "context_switch_cost": csc,
        "recovery_alignment_score": ras,
    }
