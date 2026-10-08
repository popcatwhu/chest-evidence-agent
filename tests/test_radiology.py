from chest_agent.radiology import split_findings


def test_interpretation_and_unverified_devices_are_not_patient_facts():
    text=('Near complete opacification of left hemithorax with mediastinal shift to the right, consistent with left lung collapse. '
          'The right lung is clear. Enteric tube tip is in the stomach.')
    result=split_findings(text)
    assert 'shift to the right' in result['visual_findings'][0]
    assert not any('collapse' in s for s in result['visual_findings'])
    assert result['unverified_interpretations']==['left lung collapse.']
    assert result['unverified_devices']==['Enteric tube tip is in the stomach.']


def test_single_image_does_not_prove_interval_change():
    result=split_findings('Nodular opacities are unchanged. Bilateral nodules are present.')
    assert result['visual_findings']==['Bilateral nodules are present']
    assert result['unverified_interpretations']==['Nodular opacities are unchanged.']
