const landingPage = document.getElementById('landing-page');
const appShell = document.getElementById('app-shell');
const sideMenu = document.getElementById('sideMenu');
const sideMenuBackdrop = document.getElementById('sideMenuBackdrop');
const sideMenuClose = document.querySelector('.side-menu-close');
const navToggles = document.querySelectorAll('.nav-toggle');
const navHome = document.querySelector('.side-menu-item[data-nav="home"]');

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

function showLandingPage() {
  if (!appShell || !landingPage) return;
  landingPage.classList.remove('is-hidden');
  landingPage.removeAttribute('aria-hidden');
  appShell.classList.add('hidden');
  document.body.classList.add('landing-active');
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
initSideMenu();

function initMarqueeScroller() {
  const marquee = document.querySelector('.model-marquee');
  const track = marquee?.querySelector('.model-track');
  if (!marquee || !track) return;

  // Disable marquee scroll on small screens; layout stacks in CSS.
  if (window.matchMedia('(max-width: 640px)').matches) {
    return;
  }

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

function initSideMenu() {
  if (!sideMenu || !sideMenuBackdrop) return;

  const openMenu = () => {
    sideMenu.classList.remove('hidden');
    sideMenuBackdrop.classList.remove('hidden');
    requestAnimationFrame(() => {
      sideMenu.classList.add('is-open');
    });
  };

  const closeMenu = () => {
    sideMenu.classList.remove('is-open');
    sideMenuBackdrop.classList.add('hidden');
    setTimeout(() => {
      sideMenu.classList.add('hidden');
    }, 200);
  };

  navToggles.forEach((toggle) => {
    toggle.addEventListener('click', () => {
      openMenu();
    });
  });

  if (sideMenuClose) {
    sideMenuClose.addEventListener('click', closeMenu);
  }

  sideMenuBackdrop.addEventListener('click', closeMenu);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeMenu();
    }
  });

  if (navHome) {
    navHome.addEventListener('click', () => {
      closeMenu();
      showLandingPage();
    });
  }

  window.showLandingPage = showLandingPage;
}
