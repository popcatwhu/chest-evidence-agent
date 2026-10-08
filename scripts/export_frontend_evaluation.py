"""Publish the completed evaluation snapshot to the existing static app."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export():
    record = json.loads((ROOT / 'medical_model_validation.json').read_text())
    clinical = ROOT / 'data/medical_comparison/clinical_label_review.json'
    if clinical.exists():
        review = json.loads(clinical.read_text())
        record['clinical_development'] = {key: review[key] for key in (
            'completed_inputs', 'unique_patients', 'specific_reference_label_matches_unique',
            'broad_family_matches_unique', 'clinical_accuracy_validated')}
    latest = ROOT / 'diagnostic_quality_validation.json'
    if latest.exists():
        quality = json.loads(latest.read_text())
        selected = quality['development'][quality['selected_reader']]
        record['current_workflow'] = {key:quality[key] for key in (
            'selected_reader','prospective_test_patients','prospective_test','clinical_accuracy_validated')}
        record['latest_clinical_development'] = {
            'completed_inputs':selected['inputs'],'unique_patients':selected['patients'],
            'specific_reference_label_matches_unique':selected['specific_matches'],
            'broad_family_matches_unique':selected['broad_matches'],
            'clinical_accuracy_validated':False,
        }
    full_reports = ROOT / 'full_report_quality_validation.json'
    if full_reports.exists():
        full = json.loads(full_reports.read_text())
        decision = full['development_selection']
        selected = decision['development_candidate'] if decision['clinical_contrast_enabled'] else decision['development_baseline']
        record['full_report_iteration'] = {
            'test_patients':full['frozen_test_patients'],
            'baseline':full['test_baseline']['modes'],
            'candidate':full['test_candidate']['modes']['verified'] if full['test_candidate'] else None,
            'clinical_contrast_enabled':decision['clinical_contrast_enabled'],
            'clinical_accuracy_validated':False,
        }
        record['latest_clinical_development'] = {
            'completed_inputs':selected['inputs'],'unique_patients':selected['patients'],
            'specific_reference_label_matches_unique':selected['specific_matches'],
            'broad_family_matches_unique':selected['broad_matches'],
            'clinical_accuracy_validated':False,
        }
    record['snapshot_exported_at'] = datetime.now(timezone.utc).isoformat()
    (ROOT / 'static/evaluation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    export()
