# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Scenario grid for the supply-chain anomaly dataset.

15 event types x 10 variants = 150 incidents. Numbers agree with the event
type, and each prompt is rendered in one of three formats with a different
subset of floor notes, so rows are not near-duplicates of one another.
"""

from __future__ import annotations

import json

import numpy as np

INDUSTRY_CONTEXT = {
    "pharma_cold_chain": "hospital pharmacy replenishment with a 2 to 8 Celsius label claim",
    "automotive_jit": "an engine-plant line that stops if this part is missing at the next shift change",
    "grocery_perishables": "a regional distribution centre feeding stores before a weekend promotion",
    "consumer_electronics": "a retail DC building online orders for a product launch week",
    "apparel": "a fashion DC processing a seasonal drop with short display windows",
    "industrial_parts": "a maintenance storeroom that supplies planned shutdowns at two plants",
}
INDUSTRIES = list(INDUSTRY_CONTEXT)
SEVERITIES = ["low", "medium", "high"]
STYLES = ["prose", "json", "shift_log"]
VARIANTS_PER_TYPE = 10
NOTES_PER_STYLE = {"prose": 6, "json": 3, "shift_log": 2}
SEVERITY_DWELL_SCALE = {"low": 0.6, "medium": 1.0, "high": 1.4}

# observed/expected ratio range, dwell-hours range, 30-day on-time range (None = not recorded)
PROFILES = {
    "late_shipment": ((1.0, 1.0), (18, 30), (0.62, 0.84)),
    "stockout": ((0.0, 0.12), (1, 10), (0.70, 0.95)),
    "overstock": ((1.45, 2.4), (6, 24), (0.82, 0.98)),
    "quality_reject": ((0.55, 0.9), (8, 36), (0.75, 0.96)),
    "customs_hold": ((0.0, 0.0), (24, 72), (0.82, 0.98)),
    "carrier_no_show": ((0.0, 0.0), (6, 20), None),
    "demand_spike": ((0.25, 0.6), (2, 12), (0.82, 0.98)),
    "forecast_miss": ((1.3, 1.9), (5, 16), (0.82, 0.98)),
    "cold_chain_excursion": ((1.0, 1.0), (5, 18), (0.82, 0.98)),
    "duplicate_asn": ((1.0, 1.0), (3, 14), (0.82, 0.98)),
    "lead_time_drift": ((0.7, 1.0), (20, 48), (0.48, 0.74)),
    "warehouse_mispick": ((0.4, 0.85), (2, 9), (0.82, 0.98)),
    "port_congestion": ((0.0, 0.0), (36, 96), (0.82, 0.98)),
    "moq_breach": ((0.0, 0.0), (8, 30), (0.82, 0.98)),
    "damage_on_receipt": ((0.5, 0.8), (3, 16), (0.82, 0.98)),
}
EVENT_TYPES = list(PROFILES)

# Eight floor notes per event type; each variant takes a different subset.
EVENT_NOTES = {
    "late_shipment": [
        "Dispatch still shows the trailer between the last GPS ping and the receiving gate.",
        "The appointment was missed and the yard has no matching licence plate.",
        "Store replenishment is holding the wave until this inbound is located.",
        "The carrier desk offered a revised arrival after the current shift ends.",
        "No shortage has been counted yet because the freight has not been unloaded.",
        "Planning will not rebook until the carrier confirms the tractor number.",
        "The outbound customer promise still assumes today's original arrival slot.",
        "Gate staff checked the yard twice and did not find this bill of lading.",
    ],
    "stockout": [
        "Pickers reached an empty primary location during the live wave.",
        "Safety stock in the reserve aisle is already allocated to an earlier order.",
        "The consuming site asked for an emergency transfer before the next shift.",
        "Backorders are stacking because available stock hit zero.",
        "A substitute SKU exists but it has not been approved for this customer.",
        "The inbound that could refill the location is not in receiving yet.",
        "Wave planning froze new allocations for this item.",
        "Cycle count confirmed the location is empty rather than a system mismatch.",
    ],
    "overstock": [
        "Forward pick slots are full and the surplus is sitting in the aisle.",
        "The last receipt was larger than the open forecast bucket.",
        "Overflow locations are almost full after this putaway.",
        "Planning was asked to cut the next purchase order for this item.",
        "No quality hold applies; the issue is space and excess units.",
        "The reserve aisle cannot take another pallet of this SKU.",
        "A returns lane was suggested to hold the extra units.",
        "The buyer still has a further delivery due this week.",
    ],
    "quality_reject": [
        "Incoming inspection failed and the lot tag is red.",
        "The WMS location must stay blocked until supplier quality responds.",
        "Do not allocate this lot to a customer order.",
        "Samples were pulled and the failed check is recorded on the lot.",
        "Putaway was stopped at the quarantine cage.",
        "The supplier quality contact has not sent a disposition.",
        "A second sample would not release the lot without a written waiver.",
        "The certificate of analysis does not match the measured result.",
    ],
    "customs_hold": [
        "The broker has the entry but no release timestamp.",
        "The container is not available for pickup at the terminal.",
        "Commercial invoice and packing list were requested again.",
        "No units from this entry have reached the distribution centre.",
        "A substitute that is already cleared could cover the wave.",
        "The hold reason code has not been shared with planning.",
        "Duties are not the stated problem; the entry is simply unreleased.",
        "The dray carrier will not book until customs posts a release.",
    ],
    "carrier_no_show": [
        "The booked carrier did not collect during the tender window.",
        "The trailer is loaded and sealed but still at the shipper dock.",
        "Backup capacity on this lane has not been tendered yet.",
        "The receiving site is waiting for a revised ETA.",
        "No GPS ping exists because the freight never left.",
        "The carrier scorecard should record this missed collection.",
        "Labour for the outbound door has been released.",
        "A spot quote is required before a replacement truck can be sent.",
    ],
    "demand_spike": [
        "Orders since yesterday jumped above the trailing ten-day baseline.",
        "Promotional allocations were not capped before the spike hit.",
        "Available stock cannot cover the new promise dates.",
        "Safety stock is the only pool left for the extra demand.",
        "The next firm receipt is the first chance to rebuild.",
        "Customer service is still confirming orders that planning cannot fill.",
        "The spike is in real orders, not a forecast edit.",
        "An expedite has not been raised on the inbound yet.",
    ],
    "forecast_miss": [
        "Actual movement diverged from the locked weekly forecast.",
        "The demand planner has not republished the short-horizon number.",
        "Sites that are long could rebalance toward sites that are short.",
        "Nothing in the record says the product is damaged or expired.",
        "The forecast error is the reason the slot plan no longer fits.",
        "A planner note is required before the next statistical run.",
        "Purchase orders still reflect the old forecast quantity.",
        "This is a planning miss, not a carrier failure.",
    ],
    "cold_chain_excursion": [
        "The logger left the labelled refrigerated range.",
        "Quality must disposition the lot before any customer allocation.",
        "The product remains in the quarantine cooler, not in pick.",
        "The logger file needs to stay attached to the lot record.",
        "A temperature deviation is open and unsigned.",
        "Putaway into available stock is forbidden while the deviation is open.",
        "The labelled range for this move is 2 to 8 Celsius.",
        "Warehouse staff did not discard the lot; they isolated it.",
    ],
    "duplicate_asn": [
        "Two advance ship notices point at one handling unit.",
        "Receiving against both notices would double the on-hand balance.",
        "The physical carton count still has to be confirmed.",
        "One of the notices should be cancelled or merged in the WMS.",
        "The labels on the pallet match only one of the notices.",
        "Inventory control flagged the duplicate before putaway finished.",
        "No damage is reported; the problem is the document pair.",
        "The second notice arrived after the first had already been scanned.",
    ],
    "lead_time_drift": [
        "Recent receipts from this supplier landed later than master data.",
        "The planning lead time has not been edited yet.",
        "Safety stock was sized for the old lead time.",
        "The supplier was asked for a recovery calendar and has not answered.",
        "Promises to customers still use the outdated lead time.",
        "This is a pattern across recent receipts, not a single traffic jam.",
        "Master data still shows the previous transit days.",
        "Buyers should stop quoting the old lead time on new orders.",
    ],
    "warehouse_mispick": [
        "The handling unit on the trailer does not match the order line.",
        "The outbound trailer has not departed, so the line can be corrected.",
        "A recount found a different SKU in the carton.",
        "The correct units are still in the pick location.",
        "A warehouse quality event should trigger a cycle count.",
        "The customer has not received the wrong item yet.",
        "The pack slip and the carton contents disagree.",
        "Shipping can swap the line before the door is closed.",
    ],
    "port_congestion": [
        "The vessel has not been given a berth.",
        "No container from this bill is at the distribution centre.",
        "A substitute SKU already on hand could cover part of the promise.",
        "Booking a pickup now would fail because the terminal cannot release it.",
        "The customer has not been told a revised date.",
        "The delay sits at the port, not at the carrier after discharge.",
        "Terminal appointment slots are not offered for this vessel yet.",
        "Drayage will wait for a berth notice before dispatching a driver.",
    ],
    "moq_breach": [
        "The supplier rejected the purchase order for being under the minimum.",
        "The open quantity is below the stated minimum order quantity.",
        "Raising the order to the minimum would create extra stock.",
        "A distributor that breaks packs is an alternative source.",
        "Leaving the order open will not produce a shipment.",
        "The buyer has not chosen between a larger buy and a broker.",
        "The supplier minimum is a commercial rule, not a quality hold.",
        "No freight has moved because the order was refused.",
    ],
    "damage_on_receipt": [
        "Receiving photographed crushed cartons before putaway.",
        "Damaged units must go to the claims location.",
        "Only the undamaged remainder can enter available stock.",
        "A carrier or supplier claim has not been filed yet.",
        "The photos are attached to the receipt in the WMS.",
        "Wet corrugate was noted on the pallet corner.",
        "Do not mix damaged units back into the sellable location.",
        "The claim clock starts at receipt, so the file should go out today.",
    ],
}
NOTE_OFFSETS = [0, 1, 3, 5, 2, 4]


def _metrics(event_type: str, severity: str, rng: np.random.Generator) -> dict:
    (ratio_lo, ratio_hi), (dwell_lo, dwell_hi), otd_range = PROFILES[event_type]
    expected = int(rng.integers(10, 40)) if event_type == "moq_breach" else int(rng.integers(80, 900))
    temperature = None
    if event_type == "cold_chain_excursion":
        temperature = round(float(rng.uniform(9.5, 16.0)), 1)
    elif rng.random() < 0.35 and event_type not in {"carrier_no_show", "customs_hold", "port_congestion"}:
        temperature = round(float(rng.uniform(2.4, 7.2)), 1)
    return {
        "quantity_expected": expected,
        "quantity_observed": int(expected * rng.uniform(ratio_lo, ratio_hi)),
        "dwell_hours": round(float(rng.uniform(dwell_lo, dwell_hi)) * SEVERITY_DWELL_SCALE[severity], 1),
        "temperature_c": temperature,
        "supplier_otd_30d": None if otd_range is None else round(float(rng.uniform(*otd_range)), 2),
    }


def _render(record: dict, note: str, style: str) -> str:
    temp, otd = record["temperature_c"], record["supplier_otd_30d"]
    if style == "json":
        return (
            "Incident record (control-tower JSON extract):\n"
            f"{json.dumps(record, indent=2)}\nOperator note: {note}\n"
            "Explain the anomaly. Return only the JSON object defined in your instructions."
        )
    if style == "shift_log":
        return "\n".join([
            f"SHIFT LOG {record['id']} | {record['shift']} | dock {record['dock']}",
            f"SKU {record['sku']} | lane {record['lane']} | industry {record['industry']}",
            f"FLAG {record['event_type']} | severity {record['severity']}",
            f"QTY expected {record['quantity_expected']} / observed {record['quantity_observed']}",
            f"DWELL_HOURS {record['dwell_hours']} | TEMP {'n/a' if temp is None else f'{temp}C'} | OTD_30D {'n/a' if otd is None else otd}",
            f"NOTE: {note}",
            "Write the explanation JSON only.",
        ])
    return (
        f"Please explain incident {record['id']}. SKU {record['sku']} moves on lane {record['lane']} "
        f"in {record['industry'].replace('_', ' ')}. The event label is {record['event_type'].replace('_', ' ')} "
        f"and the severity flag is {record['severity']}. We expected {record['quantity_expected']} units and the "
        f"floor counted {record['quantity_observed']}. It has dwelled {record['dwell_hours']} hours at dock "
        f"{record['dock']} during the {record['shift']} shift. "
        + ("Temperature was not recorded. " if temp is None else f"The logger reads {temp} Celsius. ")
        + ("No 30-day on-time rate is available. " if otd is None else f"The supplier's 30-day on-time rate is {otd}. ")
        + f"{note} Return only the JSON explanation."
    )


def build_scenarios(seed: int = 7) -> list[dict]:
    """Return 150 rows with example_id, event_type, industry, and user_text."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, event_type in enumerate(EVENT_TYPES):
        for v in range(VARIANTS_PER_TYPE):
            industry = INDUSTRIES[(t + v) % len(INDUSTRIES)]
            severity = SEVERITIES[(t + v) % len(SEVERITIES)]
            style = STYLES[(2 * t + v) % len(STYLES)]
            serial = t * VARIANTS_PER_TYPE + v + 1
            record = {
                "id": f"INC-{seed}-{serial:04d}",
                "event_type": event_type,
                "industry": industry,
                "severity": severity,
                "sku": f"{industry[:3].upper()}-{event_type[:4].upper()}-{1000 + serial}",
                "lane": f"{['SGN', 'SHA', 'MUN', 'RTM', 'MEM', 'NGB'][v % 6]}-{['ROT', 'LAX', 'ORD', 'FRA', 'ATL', 'LGB'][(v + t) % 6]}",
                "dock": f"D{(3 * v + t) % 40 + 1}",
                "shift": ["nights", "days", "afternoons"][v % 3],
                **_metrics(event_type, severity, rng),
            }
            bank = EVENT_NOTES[event_type]
            picked = [bank[(v + off) % len(bank)] for off in NOTE_OFFSETS[: NOTES_PER_STYLE[style]]]
            note = f"Context: {INDUSTRY_CONTEXT[industry]}. " + " ".join(picked)
            rows.append({
                "example_id": record["id"],
                "event_type": event_type,
                "industry": industry,
                "user_text": _render(record, note, style),
            })
    return rows
