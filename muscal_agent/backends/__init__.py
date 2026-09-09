"""Optional backends. Each module degrades gracefully when its deps are missing.

Nothing here is imported until a tool actually runs, so the agent can be used
for parsing / evaluation without Playwright or pyautogui installed.
"""
