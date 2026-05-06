/**
 * Sidebar Component Logic
 * Handles tab switching between Doctor and Developer sections.
 */
function switchTab(tab) {
  document.querySelectorAll('.app-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.app-section').forEach(s => s.classList.remove('active'));
  
  const doctorTab = document.querySelector('.app-tab:nth-child(1)');
  const developerTab = document.querySelector('.app-tab:nth-child(2)');
  const doctorSection = document.getElementById('section-doctor');
  const developerSection = document.getElementById('section-developer');

  if (tab === 'doctor') {
    if (doctorTab) doctorTab.classList.add('active');
    if (doctorSection) doctorSection.classList.add('active');
  } else {
    if (developerTab) developerTab.classList.add('active');
    if (developerSection) developerSection.classList.add('active');
    // Initialize Drawflow if needed (developer tab specific)
    if (typeof initDrawflow === 'function' && typeof editor === 'undefined') {
        initDrawflow();
    }
  }
}
