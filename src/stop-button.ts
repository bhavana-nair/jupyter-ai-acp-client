import { stopStreaming } from './request';

/**
 * Props for the stop streaming button.
 */
export interface IStopButtonProps {
  chatPath: string;
  personaMentionName: string | null;
}

/**
 * Creates a minimal stop-streaming button as a DOM element.
 *
 * This is a POC — no React dependency required. The button calls
 * `stopStreaming()` on click and disables itself while the request
 * is in flight.
 *
 * Usage:
 *   const btn = createStopButton({ chatPath: 'chat.chat', personaMentionName: null });
 *   someContainer.appendChild(btn);
 *   // To show/hide based on streaming state:
 *   btn.style.display = isWriting ? '' : 'none';
 */
export function createStopButton(props: IStopButtonProps): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.className = 'jp-jupyter-ai-acp-client-stopButton';
  btn.title = 'Stop streaming';
  btn.textContent = '■ Stop';

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      await stopStreaming(props.chatPath, props.personaMentionName);
    } finally {
      btn.disabled = false;
    }
  });

  return btn;
}
