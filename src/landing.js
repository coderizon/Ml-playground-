const landingPage = document.getElementById('landing-page');
const appShell = document.getElementById('app-shell');

function startApp(targetMode) {
  if (!appShell || !landingPage) return;

  if (typeof window.setAppMode === 'function' && targetMode) {
    window.setAppMode(targetMode);
  }

  landingPage.classList.add('is-hidden');
  landingPage.setAttribute('aria-hidden', 'true');
  appShell.classList.remove('hidden');
  document.body.classList.remove('landing-active');
}

if (landingPage) {
  const startButtons = landingPage.querySelectorAll('[data-mode]');

  startButtons.forEach((button) => {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      const mode = button.getAttribute('data-mode');
      const availabilityNode = button.closest('[data-available]');
      const isAvailable = !availabilityNode || availabilityNode.getAttribute('data-available') !== 'false';

      if (button.getAttribute('aria-disabled') === 'true') {
        event.preventDefault();
        return;
      }

      if (!isAvailable) {
        alert('Dieser Modus ist noch in Arbeit.');
        return;
      }

      startApp(mode);
    });
  });

  landingPage.addEventListener('keydown', (event) => {
    const isActivationKey = event.key === 'Enter' || event.key === ' ';
    if (!isActivationKey || !(event.target instanceof Element)) return;
    const modeTarget = event.target.closest('[data-mode]');
    const isNativeButton = modeTarget instanceof HTMLButtonElement;
    if (modeTarget && !isNativeButton) {
      event.preventDefault();
      modeTarget.dispatchEvent(new Event('click', { bubbles: true }));
    }
  });
}

initMarqueeScroller();

function initMarqueeScroller() {
  const marquee = document.querySelector('.model-marquee');
  const track = marquee?.querySelector('.model-track');
  if (!marquee || !track) return;

  const AUTO_SPEED = 0.02; // px per ms, deutlich langsamer
  let isDragging = false;
  let startX = 0;
  let startScroll = 0;
  let lastTime = null;
  let autoScroll = true;
  let activePointerId = null;

  const ensureLoop = () => {
    const halfWidth = track.scrollWidth / 2;
    if (halfWidth === 0) return;
    if (marquee.scrollLeft >= halfWidth) {
      marquee.scrollLeft -= halfWidth;
    } else if (marquee.scrollLeft < 0) {
      marquee.scrollLeft += halfWidth;
    }
  };

  marquee.scrollLeft = track.scrollWidth / 4;

  const step = (timestamp) => {
    if (lastTime === null) lastTime = timestamp;
    const delta = timestamp - lastTime;
    if (autoScroll && !isDragging) {
      marquee.scrollLeft += delta * AUTO_SPEED;
      ensureLoop();
    }
    lastTime = timestamp;
    requestAnimationFrame(step);
  };

  requestAnimationFrame(step);

  marquee.addEventListener('pointerdown', (event) => {
    activePointerId = event.pointerId;
    isDragging = true;
    autoScroll = false;
    startX = event.clientX;
    startScroll = marquee.scrollLeft;
    marquee.classList.add('is-dragging');
  });

  marquee.addEventListener('pointermove', (event) => {
    if (!isDragging || event.pointerId !== activePointerId) return;
    event.preventDefault();
    const delta = event.clientX - startX;
    marquee.scrollLeft = startScroll - delta;
    ensureLoop();
  });

  const endDrag = (event) => {
    if (!isDragging || (event.pointerId && event.pointerId !== activePointerId)) return;
    isDragging = false;
    activePointerId = null;
    marquee.classList.remove('is-dragging');
    autoScroll = true;
    lastTime = null;
  };

  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  window.addEventListener('pointerleave', endDrag);
}
