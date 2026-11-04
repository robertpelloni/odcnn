from playwright.sync_api import sync_playwright

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:8000", wait_until="networkidle")

        # Verify page components
        assert "ODCNN AUTOPILOT STUDIO" in page.content()
        assert page.locator("#btnSynthesize").is_visible()
        assert page.locator("#btnInfer").is_visible()
        assert page.locator("#btnPrep").is_visible()
        assert page.locator("#btnTrain").is_visible()

        # Take screenshot of the complete dashboard UI
        screenshot_path = "/app/tests/dashboard_screenshot.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot successfully saved to {screenshot_path}")
        browser.close()

if __name__ == "__main__":
    verify_dashboard()
