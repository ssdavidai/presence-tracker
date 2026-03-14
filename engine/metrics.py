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

v1.4 — Solo-meeting fix (SDI + FDI):
  Social Drain Index (SDI) and Focus Depth Index (FDI) now distinguish between
  social meetings (with other attendees) and solo calendar blocks (focus sessions,
  personal events, reminders).

  Problem: any window with in_meeting=True was treated as a social interaction.
  Solo calendar blocks — deliberate deep-work focus periods, baby-swimming classes,
  personal reminders — have attendee_count == 0 (or 1 via old gcal default) but
  were registering SDI ≈ 0.30–0.35 and FDI ≈ 0.60 due to the binary meeting flag.

  SDI fix: meeting_component (0.30 weight) now only fires for social meetings
    (meeting_attendees > 1).  A solo calendar block correctly returns SDI near 0.

  FDI fix: meeting_disruption (0.40 weight) now only fires for social meetings.
    A solo focus block no longer penalises focus depth — it IS deep focus.

  gcal.py fix: attendee_count default changed from 1 to 0 for events with no
    attendees.  The old default ("at minimum, just David") conflated David's
    presence with the presence of other participants, inflating attendee counts
    for all personal calendar events.

  Impact: solo calendar blocks now return SDI ≈ 0 and FDI ≈ 1.0 (appropriate).
    Social meetings are unaffected (>1 attendees still fires the full signal).
    This corrects a systematic overestimation of social drain for personal events.

v1.5 — Solo-meeting fix extended to CLS:
  Cognitive Load Score (CLS) now also distinguishes social meetings from solo
  calendar blocks, completing the fix started in v1.4 for SDI and FDI.

  Problem: meeting_component (0.35 weight) fired for ANY in_meeting=True window,
  giving solo focus blocks (attendees=0 or 1) CLS ≈ 0.40 — identical to a social
  meeting.  This inflated daily-average CLS on days with deliberate focus blocks,
  and misrepresented what the signal actually measures (external coordination
  overhead, not internal thinking effort).

  CLS fix: meeting_component and calendar_pressure now only fire for social meetings
    (meeting_attendees > 1).  A solo calendar block contributes only the
    physiological baseline (recovery_inverse) and any Slack activity to CLS.
    This is semantically correct: CLS captures external cognitive demand
    (meetings, interruptions, coordination) not internal focus effort.

  Impact:
    - Solo focus block: CLS ≈ 0.04–0.07 (from recovery_inverse only) → meaningful
    - Social meeting:   CLS ≈ 0.35–0.65 (full meeting signal) → unchanged
    - Daily-average CLS on focus-block days drops meaningfully, reflecting real load
    - RAS improves correctly for focused days (was penalised by inflated CLS)

  Consistency: CLS, FDI, and SDI now all use the same is_social_meeting gate.

v1.6 — Solo-meeting fix extended to CSC:
  Context Switch Cost (CSC) now also distinguishes social meetings from solo
  calendar blocks, completing the consistency fix across all five metrics.

  Problem: CSC's meeting_switch and fragmentation components fired for ANY
  in_meeting=True window.  A solo calendar block (baby-swimming class, focus
  session, personal reminder) was getting CSC ≈ 0.25–0.60 depending on duration,
  despite zero context-switching overhead.

  The v1.4/v1.5 series fixed SDI, FDI, and CLS but left CSC incorrectly
  treating solo blocks as context-switching events.  A dedicated focus block
  (in_meeting=True, attendees ≤ 1) is the opposite of a context switch — it
  is sustained, undivided engagement.

  CSC fix: meeting_switch and fragmentation now only fire for social meetings
    (meeting_attendees > 1).  A solo calendar block contributes only the
    Slack-channel and RescueTime app-switch signals to CSC.
    Semantics: CSC measures *external* scheduling fragmentation (back-to-back
    short meetings with different people), not internal duration of focus.

  API change: context_switch_cost() now accepts meeting_attendees (default 0)
    to enable the is_social_meeting gate.  Existing callers without this kwarg
    default to attendees=0, which is the safe conservative direction (no false
    CSC inflation for unknown events).

  Impact:
    - Solo focus block (any duration): CSC = 0.0 (from Slack/RT signals only)
    - Social meeting, long (≥30 min):  CSC ≈ 0.20 (unchanged)
    - Social meeting, short (<30 min): CSC ≈ 0.45–0.55 (unchanged)
    - CSC is now semantically consistent with CLS, FDI, and SDI across all
      four meeting-sensitive metrics.

v2.0 — Omi transcript integration:
  cognitive_load_score(), social_drain_index(), and focus_depth_index() now
  accept optional Omi transcript signals:

  CLS: spoken conversation in a window is a direct cognitive-load signal.
    Processing live speech, formulating verbal responses, and tracking
    multi-participant discussions are all high-demand tasks.
    omi_conversation_active and omi_word_count upgrade the Slack-message
    proxy with a real spoken-word signal.

  SDI: spoken conversation is the most direct social-energy signal possible
    — far more demanding than text messages.  omi_speech_seconds drives
    the Omi component of SDI proportionally to actual speaking time.

  FDI: active conversation (Omi) is a disruption to deep focus, just as
    social meetings are.  When Omi signals a conversation, the disruption
    component increases proportionally to speech ratio.

  All three functions add optional omi_* kwargs that default to None / 0 so
  existing callers without Omi data continue to work unchanged.  Omi signals
  blend into existing weights rather than replacing them, so the metric
  degrades gracefully to baseline when Omi is unavailable.

  Impact:
    - Windows with active spoken conversation: CLS +0.05–0.15, FDI −0.05–0.15,
      SDI +0.05–0.20 (proportional to word count / speech duration)
    - Windows without Omi: identical to previous behaviour (all changes backward-compatible)
    - sources_available gains "omi" when transcript data is present for that window
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
    omi_conversation_active: bool = False,
    omi_word_count: int = 0,
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
        omi_conversation_active: bool — at least one Omi transcript session
            started in this window.  Spoken conversation = high cognitive demand.
            (v2.0 — Omi transcript integration)
        omi_word_count: int — total words spoken across Omi sessions in window.
            More words = higher engagement = higher CLS.  Saturates at 500 words
            per 15-min window (≈33 words/min = dense conversation).

    v1.5 — Solo-meeting fix:
        meeting_component and calendar_pressure only fire for social meetings
        (meeting_attendees > 1).  A solo calendar block (focus session,
        personal event, attendees=0/1) does NOT generate the coordination
        and participation overhead that meeting_component captures.
        Consistent with the v1.4 solo-meeting fix applied to FDI and SDI.

    v2.0 — Omi integration:
        When omi_conversation_active is True, an Omi spoken-conversation signal
        is blended into CLS.  The blend weight is 0.10 (10%) applied on top of
        the base formula:
          omi_component = 0.5 + 0.5 * word_density  (0.5 baseline + word-count boost)
          cls = 0.90 * base_cls + 0.10 * omi_component
        This is conservative — Omi adds a nudge rather than dominating the metric.
        A quiet-but-active Omi window (few words) raises CLS by ~0.05 over baseline.
        A word-dense Omi window (>500 words) raises CLS by ~0.08–0.10.
    """
    # Component: meeting density
    # v1.5 solo-meeting fix: only fire for social meetings (other attendees present).
    # A solo calendar block (focus session, personal event, attendees=0/1) does NOT
    # generate external coordination overhead — it IS protected focus time.
    # CLS captures external cognitive demand (coordination, meetings, interruptions),
    # not internal thinking effort.  Solo blocks contribute only the physiological
    # baseline (recovery_inverse) and any Slack interruptions.
    is_social_meeting = in_meeting and meeting_attendees > 1
    meeting_component = 1.0 if is_social_meeting else 0.0

    # Component: calendar pressure
    # Normalized by max expected attendees (10).
    # Only fires for social meetings (attendee_count only meaningful when social).
    calendar_pressure = _norm(meeting_attendees if is_social_meeting else 0, max_val=10)

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
        base_cls = 0.75 * base_cls + 0.25 * rt_demand

    # v2.0: Omi spoken-conversation signal
    # Live spoken conversation is cognitively demanding: David must process speech,
    # formulate verbal responses, and track conversation threads in real time.
    # Word-count density captures engagement depth: a 5-word exchange is light;
    # a 500-word back-and-forth is intense.
    # Saturates at 500 words/window (≈33 wpm, dense conversation for 15 min).
    if omi_conversation_active:
        word_density = _norm(omi_word_count, max_val=500)
        # omi_component: 0.5 baseline (just being in conversation) + 0.5 * word density
        omi_component = 0.5 + 0.5 * word_density
        cls = 0.90 * base_cls + 0.10 * omi_component
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
    meeting_attendees: int = 0,
    omi_conversation_active: bool = False,
    omi_speech_ratio: float = 0.0,
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
        meeting_attendees: number of participants in the active meeting
            (v1.4).  When provided, used to distinguish social meetings
            (disrupted focus) from solo focus blocks (protected time).
        omi_conversation_active: bool — at least one Omi session in this window.
            Active spoken conversation disrupts deep focus.
            (v2.0 — Omi transcript integration)
        omi_speech_ratio: float — speech_seconds / audio_seconds (0.0–1.0).
            Higher ratio = more active conversation = more disruption.

    v1.2: When RescueTime data is present (rt_app_switches is not None
    and rt_active_seconds >= 60), real app-switch counts replace the
    Slack-channels proxy for context switching, improving precision of
    the FDI signal significantly.

    v1.4 — Solo-meeting fix:
      A calendar block with no other attendees (meeting_attendees ≤ 1) is a
      scheduled focus session, not a social meeting.  Social meetings interrupt
      deep work; solo blocks protect it.  meeting_disruption now only fires for
      social meetings (meeting_attendees > 1), so deliberate focus-block scheduling
      is no longer penalised as a disruption in the FDI signal.

    v2.0 — Omi integration:
      Active spoken conversation disrupts deep focus.  When Omi reports a
      conversation in this window, an omi_disruption component is blended in:
        omi_disruption = 0.5 + 0.5 * speech_ratio
        (0.5 baseline for being in any conversation + 0.5 * density)
      FDI = 0.90 * base_fdi - 0.10 * omi_disruption  (clamped to [0, 1])
      A heavily spoken window (speech_ratio ≈ 1.0) reduces FDI by ≈ 0.09–0.10.
      A light Omi session (speech_ratio ≈ 0.3) reduces FDI by ≈ 0.06.
    """
    # Social meetings (with other attendees) break focus.
    # Solo calendar blocks (focus sessions, personal events) do not.
    is_social_meeting = in_meeting and meeting_attendees > 1
    meeting_disruption = 1.0 if is_social_meeting else 0.0

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

    # v2.0: Omi spoken-conversation disruption signal.
    # Active conversation breaks the focus state regardless of meeting status.
    # speech_ratio (0–1) captures how dense the conversation was:
    # low ratio = brief/sporadic speech; high ratio = sustained conversation.
    if omi_conversation_active:
        omi_disruption = 0.5 + 0.5 * _clamp(omi_speech_ratio)
        # Blend: Omi adds 10% disruption weight on top of base calculation
        fdi = _clamp(1.0 - disruption) * 0.90 - 0.10 * omi_disruption + 0.10
        # Simplification: fdi = 0.90*(1-disruption) + 0.10*(1-omi_disruption)
        fdi = 0.90 * (1.0 - disruption) + 0.10 * (1.0 - omi_disruption)
    else:
        fdi = 1.0 - disruption

    return round(_clamp(fdi), 4)


# ─── SDI: Social Drain Index ─────────────────────────────────────────────────

def social_drain_index(
    in_meeting: bool,
    meeting_attendees: int,
    slack_messages_sent: int,
    slack_messages_received: int,
    omi_conversation_active: bool = False,
    omi_speech_seconds: float = 0.0,
) -> float:
    """
    Social Drain Index — how much social energy was expended?

    Range: 0.0 (isolated/quiet) to 1.0 (maximum social engagement)

    Higher values indicate more social interactions that may drain energy.

    v1.4 — Solo-meeting fix:
      meeting_component now only fires for social meetings (meeting_attendees > 1).
      A solo calendar block (focus session, personal reminder, event with no other
      attendees) has meeting_attendees == 0 or 1 and carries zero social drain —
      it is deliberate protected time, not a social interaction.

      Previously meeting_component = 1.0 for ANY in_meeting window, causing solo
      blocks to register SDI ≈ 0.30–0.35 despite zero social interaction.
      This inflated weekly social drain stats for focus-protective scheduling.

      The fix: is_social_meeting = in_meeting AND meeting_attendees > 1
      Solo focus blocks (attendees ≤ 1) correctly return SDI near 0.

    v2.0 — Omi integration:
      Spoken conversation is the most direct social-energy signal possible.
      A 10-minute voice conversation drains far more social energy than
      10 Slack messages.  omi_speech_seconds captures actual speaking time:

      omi_sdi_component = speech_seconds / 900  (900s = full 15-min window)
      When omi is active: SDI = 0.85 * base_sdi + 0.15 * omi_component

      The 15% Omi blend is conservative but meaningful:
      - 900s speech (full window active) adds ~0.15 to SDI
      - 180s speech (typical 3-min Omi session) adds ~0.03 to SDI
      - No Omi: identical to baseline (backward-compatible)
    """
    # A meeting is "social" only when there is at least one other participant.
    # Solo focus blocks, personal events, and calendar reminders have 0–1 attendee
    # and should not contribute to social drain.
    is_social_meeting = in_meeting and meeting_attendees > 1

    # Large social meetings drain more energy (normalised by max expected = 10)
    attendee_component = _norm(meeting_attendees if is_social_meeting else 0, max_val=10)

    # Binary signal: are we in a social meeting at all?
    meeting_component = 1.0 if is_social_meeting else 0.0

    # Sent messages (active communication) vs received (passive)
    total_slack = slack_messages_sent + slack_messages_received
    if total_slack > 0:
        sent_ratio = slack_messages_sent / total_slack
    else:
        sent_ratio = 0.0

    base_sdi = (
        0.50 * attendee_component +
        0.30 * meeting_component +
        0.20 * sent_ratio
    )

    # v2.0: Omi spoken-conversation signal
    # Spoken conversation is inherently social — it requires real-time engagement
    # with another person and is more draining than text communication.
    # speech_seconds saturates at 900 (full 15-min window of continuous speech).
    if omi_conversation_active and omi_speech_seconds > 0:
        omi_component = _norm(omi_speech_seconds, max_val=900.0)
        sdi = 0.85 * base_sdi + 0.15 * omi_component
    else:
        sdi = base_sdi

    return round(_clamp(sdi), 4)


# ─── CSC: Context Switch Cost ────────────────────────────────────────────────

def context_switch_cost(
    in_meeting: bool,
    meeting_duration_minutes: int,
    slack_channels_active: int,
    is_short_meeting: bool = False,
    rt_app_switches: Optional[int] = None,
    rt_active_seconds: int = 0,
    meeting_attendees: int = 0,
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

    v1.6 — Solo-meeting fix:
      meeting_switch and fragmentation now only fire for social meetings
      (meeting_attendees > 1).  A solo calendar block (dedicated focus session,
      baby-swimming class, personal event) is the opposite of context switching —
      it is sustained, undivided engagement.  Treating solo blocks as context-
      switch events inflated CSC for focus-protective scheduling.

      Semantics: CSC measures external scheduling fragmentation (back-to-back
      short meetings with different people), not internal focus duration.
      meeting_attendees defaults to 0 so callers without the kwarg default to
      the conservative direction (no false CSC inflation for unknown events).
    """
    # v1.6: Only social meetings (with other participants) incur context-switch cost.
    # A solo block is deliberate sustained focus — the opposite of fragmentation.
    is_social_meeting = in_meeting and meeting_attendees > 1

    # Short *social* meetings are the costliest context switches — they interrupt
    # deep work without providing sufficient sustained engagement.
    if is_social_meeting:
        meeting_switch = 0.5 + (0.5 if is_short_meeting else 0.0)
    else:
        meeting_switch = 0.0

    # Cross-channel Slack activity (applies regardless of meeting type)
    channel_switch = _norm(slack_channels_active, max_val=5)

    # Calendar fragmentation: very short *social* meetings (<15 min)
    # Solo blocks of any duration do not fragment the day; they protect it.
    fragmentation = 1.0 if (is_social_meeting and meeting_duration_minutes < 15) else (
        0.5 if (is_social_meeting and meeting_duration_minutes < 30) else 0.0
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

    Optional (v2.0):
    - omi: {conversation_active, word_count, speech_seconds, audio_seconds,
            sessions_count, speech_ratio}
      When present, Omi spoken-conversation signals upgrade CLS, SDI, and FDI
      with real verbal interaction data.

    v1.1: hrv_rmssd_milli and sleep_performance are now forwarded to CLS
    and RAS so the physiological readiness composite is fully utilised.

    v1.2: rescuetime sub-dict is now extracted and forwarded to CLS, FDI,
    and CSC so that real behavioral signals (app_switches, productivity_score)
    are used when available, replacing the Slack-channel proxies.

    v1.6: meeting_attendees is now forwarded to CSC so that the solo-meeting
    fix (is_social_meeting gate) applies consistently across all four meeting-
    sensitive metrics: CLS, FDI, SDI, and CSC.

    v2.0: omi sub-dict is extracted and forwarded to CLS, FDI, and SDI so that
    real spoken-conversation signals upgrade those three metrics when Omi
    transcript data is available.
    """
    cal = window_data.get("calendar", {})
    whoop = window_data.get("whoop", {})
    slack = window_data.get("slack", {})
    rt = window_data.get("rescuetime") or {}  # Optional — None-safe
    omi = window_data.get("omi") or {}        # Optional — None-safe (v2.0)

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

    # Omi spoken-conversation signals (v2.0)
    omi_conversation_active: bool = bool(omi.get("conversation_active", False)) if omi else False
    omi_word_count: int = int(omi.get("word_count", 0)) if omi else 0
    omi_speech_seconds: float = float(omi.get("speech_seconds", 0.0)) if omi else 0.0
    omi_speech_ratio: float = float(omi.get("speech_ratio", 0.0)) if omi else 0.0

    cls = cognitive_load_score(
        in_meeting=in_meeting,
        meeting_attendees=meeting_attendees,
        slack_messages_received=slack_received,
        recovery_score=recovery_score,
        hrv_rmssd_milli=hrv_rmssd_milli,
        sleep_performance=sleep_performance,
        rt_productivity_score=rt_productivity_score,
        rt_active_seconds=rt_active_seconds,
        omi_conversation_active=omi_conversation_active,  # v2.0
        omi_word_count=omi_word_count,                    # v2.0
    )

    fdi = focus_depth_index(
        in_meeting=in_meeting,
        slack_messages_received=slack_received,
        context_switches=slack_channels,  # Proxy when RescueTime unavailable
        rt_app_switches=rt_app_switches,
        rt_active_seconds=rt_active_seconds,
        rt_productivity_score=rt_productivity_score,
        meeting_attendees=meeting_attendees,              # v1.4: solo-block awareness
        omi_conversation_active=omi_conversation_active,  # v2.0
        omi_speech_ratio=omi_speech_ratio,                # v2.0
    )

    sdi = social_drain_index(
        in_meeting=in_meeting,
        meeting_attendees=meeting_attendees,
        slack_messages_sent=slack_sent,
        slack_messages_received=slack_received,
        omi_conversation_active=omi_conversation_active,  # v2.0
        omi_speech_seconds=omi_speech_seconds,            # v2.0
    )

    csc = context_switch_cost(
        in_meeting=in_meeting,
        meeting_duration_minutes=meeting_duration,
        slack_channels_active=slack_channels,
        is_short_meeting=(meeting_duration < 30) if in_meeting else False,
        rt_app_switches=rt_app_switches,
        rt_active_seconds=rt_active_seconds,
        meeting_attendees=meeting_attendees,  # v1.6: solo-block awareness
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
