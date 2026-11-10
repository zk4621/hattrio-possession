from possession.io_atd import kickoff_tag_frames

XML = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>447.4</start><end>452.0</end><code> SET PIECES </code>
<label><group>Type</group><text> Kick Off (Start) </text></label>
<label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>4549.0</start><end>4554.0</end><code>SET PIECES</code>
<label><group>Type</group><text>Kick Off (Start)</text></label>
<label><group>Half</group><text> 2nd Half </text></label></instance>
<instance><ID>3</ID><start>100.0</start><end>110.0</end><code>BUILD UP</code>
<label><group>Half</group><text>1st Half</text></label></instance>
</ALL_INSTANCES></file>"""


def test_kickoff_tags_tolerate_whitespace(tmp_path, monkeypatch):
    (tmp_path / "tactical").mkdir()
    (tmp_path / "tactical" / "BAR-ATH_tactical_smart_tagging.xml").write_text(XML)
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path))
    assert kickoff_tag_frames() == {1: 11185, 2: 113725}
