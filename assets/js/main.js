/**
 * Lorenzo Sciarretta — Tech Minimal Interaction
 * Staggered Scroll Reveal + Theme Toggling
 */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initRevealAnimations();
});

/**
 * Theme Toggling
 */
function initTheme() {
  const themeToggle = document.getElementById('theme-toggle');
  const html = document.documentElement;
  
  // Load saved theme or default to dark
  const savedTheme = localStorage.getItem('theme') || 'dark';
  html.setAttribute('data-theme', savedTheme);
  
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const currentTheme = html.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      html.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
    });
  }
}

/**
 * Scroll Reveal Animations (Intersection Observer)
 */
function initRevealAnimations() {
  const revealElements = document.querySelectorAll('.reveal');
  
  const observerOptions = {
    threshold: 0.15,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        // Add a slight delay for staggered effect if multiple elements hit at once
        // This is a simple version; for more complex ones we can check siblings
        entry.target.classList.add('in-view');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);
  
  revealElements.forEach((el) => {
    observer.observe(el);
  });
}
