"""
Presence Tracker — Metric Computation Engine

Computes the 5 derived metrics for each 15-minute observation window:
- Cognitive Load Score (CLS)
- Focus Depth Index (FDI)
- Social Drain Index (SDI)
- Context Switch Cost (CSC)
- Recovery Alignment Score (RAS)

All metrics are normalized to [0.0, 1.0].
"""

from typing import Optional

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


# ─── CLS: Cognitive Load Score ───────────────────────────────────────────────

def cognitive_load_score(
    in_meeting: bool,
    meeting_attendees: int,
    slack_messages_received: int,
    recovery_score: Optional[float],
    window_duration_minutes: int = 15,
) -> float:
    """
    Cognitive Load Score — how mentally demanding was this window?

    Range: 0.0 (completely idle/recovered) to 1.0 (maximum load)

    Inputs:
        in_meeting: whether a calendar event was active
        meeting_attendees: number of participants in active meeting
        slack_messages_received: incoming Slack messages in this window
        recovery_score: WHOOP recovery (0-100), None if unavailable
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

    # Component: Recovery inverse (low recovery = higher load threshold)
    # If recovery is unknown, assume neutral (0.5 inverse)
    if recovery_score is not None:
        recovery_inverse = 1.0 - (recovery_score / 100.0)
    else:
        recovery_inverse = 0.5

    # Weighted average
    cls = (
        0.35 * meeting_component +
        0.20 * calendar_pressure +
        0.25 * slack_component +
        0.20 * recovery_inverse
    )

    return round(_clamp(cls), 4)


# ─── FDI: Focus Depth Index ──────────────────────────────────────────────────

def focus_depth_index(
    in_meeting: bool,
    slack_messages_received: int,
    context_switches: int = 0,
) -> float:
    """
    Focus Depth Index — how deep was the focus in this window?

    Range: 0.0 (completely fragmented) to 1.0 (deep, uninterrupted focus)

    Inputs:
        in_meeting: whether a calendar event was active
        slack_messages_received: incoming messages
        context_switches: app switches (v2: from RescueTime, default 0)
    """
    # Meetings break focus
    meeting_disruption = 1.0 if in_meeting else 0.0

    # Slack interruptions
    slack_disruption = _norm(slack_messages_received, max_val=30)

    # Context switches (v2: RescueTime; for now derived from slack channels)
    switch_disruption = _norm(context_switches, max_val=20)

    # FDI = 1 - disruption score
    disruption = (
        0.40 * meeting_disruption +
        0.40 * slack_disruption +
        0.20 * switch_disruption
    )

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
) -> float:
    """
    Context Switch Cost — fragmentation penalty for this window.

    Range: 0.0 (no switching, sustained mode) to 1.0 (maximum fragmentation)

    Short meetings (<30 min) are costlier than long ones because they force
    rapid context switches without settling into deep work.
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
) -> float:
    """
    Recovery Alignment Score — is your activity level appropriate for your physiology?

    Range: 0.0 (badly misaligned) to 1.0 (perfectly aligned)

    High recovery + low CLS = aligned (resting on a rest day) → high RAS
    High recovery + high CLS = aligned (working hard when ready) → high RAS
    Low recovery + low CLS = semi-aligned (resting when tired) → medium RAS
    Low recovery + high CLS = misaligned (pushing hard when depleted) → low RAS
    """
    if recovery_score is None:
        return 0.5  # Unknown, assume neutral

    recovery_norm = recovery_score / 100.0

    # Best case: high recovery lets you handle high CLS
    # Worst case: low recovery + high CLS
    # RAS = recovery × (1 - CLS) + recovery × CLS × bonus
    # Simplified: reward when recovery permits the load
    capacity_used = cls
    capacity_available = recovery_norm

    if capacity_available >= capacity_used:
        # Within capacity: full alignment
        ras = 1.0 - (0.3 * (1.0 - capacity_available))  # slight penalty for underutilizing high recovery
    else:
        # Over capacity: penalty proportional to deficit
        overload = capacity_used - capacity_available
        ras = 1.0 - overload

    return round(_clamp(ras), 4)


# ─── Master metric computation ───────────────────────────────────────────────

def compute_metrics(window_data: dict) -> dict:
    """
    Compute all 5 metrics for a single window.

    window_data must contain:
    - calendar: {in_meeting, meeting_attendees, meeting_duration_minutes}
    - whoop: {recovery_score}
    - slack: {messages_sent, messages_received, channels_active}
    """
    cal = window_data.get("calendar", {})
    whoop = window_data.get("whoop", {})
    slack = window_data.get("slack", {})

    in_meeting = cal.get("in_meeting", False)
    meeting_attendees = cal.get("meeting_attendees", 0)
    meeting_duration = cal.get("meeting_duration_minutes", 0)
    recovery_score = whoop.get("recovery_score")
    slack_sent = slack.get("messages_sent", 0)
    slack_received = slack.get("messages_received", 0)
    slack_channels = slack.get("channels_active", 0)

    cls = cognitive_load_score(
        in_meeting=in_meeting,
        meeting_attendees=meeting_attendees,
        slack_messages_received=slack_received,
        recovery_score=recovery_score,
    )

    fdi = focus_depth_index(
        in_meeting=in_meeting,
        slack_messages_received=slack_received,
        context_switches=slack_channels,  # Proxy until RescueTime is available
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
    )

    ras = recovery_alignment_score(
        recovery_score=recovery_score,
        cls=cls,
    )

    return {
        "cognitive_load_score": cls,
        "focus_depth_index": fdi,
        "social_drain_index": sdi,
        "context_switch_cost": csc,
        "recovery_alignment_score": ras,
    }
