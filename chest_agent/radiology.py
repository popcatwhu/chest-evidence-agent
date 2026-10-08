"""Separate a generated report's observations from its unverified interpretations/devices."""

import re


def split_findings(text):
    visual = []
    interpretations = []
    devices = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if re.search(
            r"\b(tube|catheter|port|pacer|pacemaker|tip|clips)\b", sentence, re.I
        ):
            devices.append(sentence)
            continue
        if re.search(r"\b(stable|unchanged|compared|interval)\b", sentence, re.I):
            interpretations.append(sentence)
            continue
        parts = re.split(r",?\s+consistent with\s+", sentence, flags=re.I, maxsplit=1)
        visual.append(parts[0].rstrip(", ."))
        if len(parts) > 1:
            interpretations.append(parts[1])
    return {
        "visual_findings": visual,
        "unverified_interpretations": interpretations,
        "unverified_devices": devices,
        "raw_report": text,
    }
