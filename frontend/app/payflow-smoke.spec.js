import { test, expect } from '@playwright/test';

const viewports = [
  ['desktop', { width: 1440, height: 1000 }],
  ['mobile', { width: 390, height: 844 }],
];

for (const [name, viewport] of viewports) {
  test(`PayFlow dashboard ${name}`, async ({ page }) => {
    const consoleErrors = [];
    const failedRequests = [];

    page.on('console', (message) => {
      if (message.type() === 'error') {
        consoleErrors.push(message.text());
      }
    });

    page.on('requestfailed', (request) => {
      if (!request.url().includes('/api/v1/stream/events')) {
        failedRequests.push(`${request.method()} ${request.url()}`);
      }
    });

    await page.setViewportSize(viewport);
    await page.goto('http://127.0.0.1:8010/app', {
      waitUntil: 'domcontentloaded',
      timeout: 60000,
    });
    await page.waitForSelector('#root', { timeout: 15000 });
    await page.waitForTimeout(8000);

    const metrics = await page.evaluate(() => ({
      rootChildren: document.querySelector('#root')?.children.length || 0,
      bodyTextLength: document.body.innerText.length,
      horizontalOverflow:
        document.documentElement.scrollWidth >
        document.documentElement.clientWidth + 2,
      controls: document.querySelectorAll('button').length,
      visuals: document.querySelectorAll('canvas, svg').length,
    }));

    expect(metrics.rootChildren).toBeGreaterThan(0);
    expect(metrics.bodyTextLength).toBeGreaterThan(500);
    expect(metrics.controls).toBeGreaterThan(0);
    expect(metrics.visuals).toBeGreaterThan(0);
    expect(metrics.horizontalOverflow).toBe(false);
    expect(consoleErrors).toEqual([]);
    expect(failedRequests).toEqual([]);
  });
}
