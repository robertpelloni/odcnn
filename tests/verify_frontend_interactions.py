import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 2200})
        page = context.new_page()

        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        print("Navigating to http://127.0.0.1:8000...")
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        time.sleep(2)

        # Verify waveform container and canvas exist
        waveform_container = page.locator("#waveformContainer")
        assert waveform_container.is_visible(), "Waveform container should be visible"

        waveform_canvas = page.locator("#waveformCanvas")
        assert waveform_canvas.is_visible(), "Waveform canvas should be visible"

        # Verify waveform metadata values populated
        track_name = page.locator("#waveTrackName").inner_text()
        print(f"Loaded waveform track name: {track_name}")
        assert len(track_name) > 0, "Track name should not be empty"

        note_count = page.locator("#waveNoteCount").inner_text()
        print(f"Loaded note count: {note_count}")

        # Verify spectrogram container and canvas exist
        spectro_container = page.locator("#spectrogramContainer")
        assert spectro_container.is_visible(), "Spectrogram container should be visible"

        spectro_canvas = page.locator("#spectrogramCanvas")
        assert spectro_canvas.is_visible(), "Spectrogram canvas should be visible"

        # Wait for spectrogram ready status
        page.wait_for_selector("#spectroStatusBadge:has-text('Spectrogram Ready')", timeout=10000)

        # Click inside waveform container to test scrubber click handling
        box = waveform_container.bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
        time.sleep(1)

        # Also click inside spectrogram container to test spectral scrub
        sbox = spectro_container.bounding_box()
        assert sbox is not None
        page.mouse.click(sbox["x"] + sbox["width"] * 0.6, sbox["y"] + sbox["height"] * 0.5)
        time.sleep(1)

        scrubber_text = page.locator("#waveScrubberTime").inner_text()
        print(f"Scrubber time text after click: {scrubber_text}")
        assert "Playback:" in scrubber_text

        # Verify Live Mic button exists
        mic_btn = page.locator("#btnToggleMic")
        assert mic_btn.is_visible()

        # Test Model Architecture Selector (CNN vs CRNN)
        arch_select = page.locator("#inferArch")
        assert arch_select.is_visible()
        arch_select.select_option("crnn")
        assert arch_select.input_value() == "crnn"
        print("Successfully selected CRNN architecture in studio")

        # Verify BPM Tap Tempo & Beat Grid Snapping Tool card
        bpm_card = page.locator("#bpmQuantizeCard")
        assert bpm_card.is_visible(), "BPM Quantize card should be visible"

        # Test Tap Tempo button clicks
        tap_btn = page.locator("#btnTapTempo")
        assert tap_btn.is_visible(), "Tap tempo button should be visible"
        tap_btn.click()
        time.sleep(0.5)
        tap_btn.click()
        time.sleep(0.5)
        tap_btn.click()
        time.sleep(0.5)

        taps_label = page.locator("#tapCountLabel").inner_text()
        print(f"Tap count label: {taps_label}")
        assert "3 taps" in taps_label

        bpm_val = page.locator("#displayBpm").inner_text()
        print(f"Calculated BPM: {bpm_val}")
        assert float(bpm_val) > 0.0

        # Test Quantize Charts button
        quantize_btn = page.locator("#btnQuantizeChart")
        assert quantize_btn.is_visible()
        quantize_btn.click()
        page.wait_for_selector("#quantizeResultBadge:has-text('Quantized')", timeout=10000)
        badge_text = page.locator("#quantizeResultBadge").inner_text()
        print(f"Quantize result: {badge_text}")
        assert "Quantized:" in badge_text

        page.screenshot(path="/home/jules/verification/verification.png", full_page=True)
        print("Screenshot saved to /home/jules/verification/verification.png")

        browser.close()
        if errors:
            print(f"Encountered page errors: {errors}")
            raise RuntimeError(f"Frontend errors occurred: {errors}")
        print("Frontend interaction verification completed with 0 errors!")

if __name__ == "__main__":
    run()
