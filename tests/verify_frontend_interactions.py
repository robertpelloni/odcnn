import sys
from playwright.sync_api import sync_playwright

def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.on("pageerror", lambda err: errors.append(f"PageError: {err}"))
        page.on("console", lambda msg: errors.append(f"ConsoleError: {msg.text}") if msg.type == "error" else None)

        page.goto("http://localhost:8000", wait_until="networkidle")

        # Verify page title
        assert "ODCNN • Musical Onset Detection" in page.title()

        # Check for upload button and stream simulate button
        btn_upload = page.locator("button:has-text('Upload Track')")
        assert btn_upload.is_visible()

        btn_stream = page.locator("button:has-text('Simulate Stream')")
        assert btn_stream.is_visible()

        # Check dialog alert when clicking simulate stream
        page.once("dialog", lambda dialog: dialog.accept())
        btn_stream.click()
        page.wait_for_timeout(1000)

        # Check logs button
        btn_logs = page.locator("button:has-text('Refresh Logs')")
        btn_logs.click()
        page.wait_for_timeout(1000)

        # Screenshot
        page.screenshot(path="/home/jules/verification/verification.png", full_page=True)
        browser.close()

    if errors:
        print("Frontend Errors encountered:", errors)
        sys.exit(1)
    else:
        print("Frontend interactions verified with 0 errors!")

if __name__ == "__main__":
    main()
