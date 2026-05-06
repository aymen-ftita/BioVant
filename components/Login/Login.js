window.Login = {
  initialized: false,
  init() {
    if (this.initialized) return;
    this.avatar = document.getElementById('doctor-avatar');
    if (!this.avatar) return;

    this.pupilL = document.getElementById('pupil-l');
    this.pupilR = document.getElementById('pupil-r');
    this.emailInput = document.getElementById('email');
    this.passwordInput = document.getElementById('password');
    this.togglePassBtn = document.getElementById('toggle-password');
    this.measureSpan = document.getElementById('email-cursor-measure');
    
    this.eyeCenterL = { x: 75, y: 85 };
    this.eyeCenterR = { x: 125, y: 85 };
    this.maxEyeMove = 6;

    this.bindEvents();
    this.updateEyePosition({ clientX: window.innerWidth / 2, clientY: window.innerHeight / 2 });
    this.initialized = true;
  },

  bindEvents() {
    // 3: the eyes of the docotr always follow the cusor in browser
    document.addEventListener('mousemove', (e) => {
      if (document.activeElement !== this.emailInput) {
        this.updateEyePosition(e);
      }
    });

    // 4: when the user click on the email field the eyes will follow the character input cursor
    this.emailInput.addEventListener('input', () => this.followInputCursor());
    this.emailInput.addEventListener('click', () => this.followInputCursor());
    this.emailInput.addEventListener('focus', () => this.followInputCursor());

    // 5: when the user click on the password field and while typing... hide eyes
    this.passwordInput.addEventListener('focus', () => this.hideEyes());
    this.passwordInput.addEventListener('input', () => this.hideEyes());
    this.passwordInput.addEventListener('blur', () => this.resetDoctor());

    // 6: show password logic
    this.togglePassBtn.addEventListener('click', () => this.togglePassword());
  },

  updateEyePosition(e) {
    if (this.avatar.classList.contains('covering-eyes')) return;

    const box = this.avatar.getBoundingClientRect();
    const avatarCenterX = box.left + box.width / 2;
    const avatarCenterY = box.top + box.height / 2;

    const angle = Math.atan2(e.clientY - avatarCenterY, e.clientX - avatarCenterX);
    const dist = Math.min(this.maxEyeMove, Math.hypot(e.clientX - avatarCenterX, e.clientY - avatarCenterY) / 50);

    const moveX = Math.cos(angle) * dist;
    const moveY = Math.sin(angle) * dist;

    this.pupilL.setAttribute('transform', `translate(${moveX}, ${moveY})`);
    this.pupilR.setAttribute('transform', `translate(${moveX}, ${moveY})`);
  },

  followInputCursor() {
    this.avatar.classList.remove('covering-eyes', 'peeking-r');
    
    // Measure cursor position
    const text = this.emailInput.value.substring(0, this.emailInput.selectionStart);
    this.measureSpan.textContent = text;
    
    // Calculate a virtual "target" point based on input position
    const inputRect = this.emailInput.getBoundingClientRect();
    const textWidth = this.measureSpan.offsetWidth;
    const targetX = inputRect.left + 16 + textWidth;
    const targetY = inputRect.top + inputRect.height / 2;

    this.updateEyePosition({ clientX: targetX, clientY: targetY });
  },

  hideEyes() {
    this.avatar.classList.remove('peeking-r');
    this.avatar.classList.add('covering-eyes');
    // Look down slightly even if covered
    this.pupilL.setAttribute('transform', `translate(0, 3)`);
    this.pupilR.setAttribute('transform', `translate(0, 3)`);
  },

  togglePassword() {
    const isPassword = this.passwordInput.type === 'password';
    this.passwordInput.type = isPassword ? 'text' : 'password';
    
    if (isPassword) {
      // Show password -> Peek
      this.avatar.classList.remove('covering-eyes');
      this.avatar.classList.add('peeking-r');
      // Look at the cursor
      const inputRect = this.passwordInput.getBoundingClientRect();
      this.updateEyePosition({ clientX: inputRect.right, clientY: inputRect.top + inputRect.height/2 });
    } else {
      // Hide password -> Cover again
      this.hideEyes();
    }
  },

  resetDoctor() {
    this.avatar.classList.remove('covering-eyes', 'peeking-r');
  }
};

// Auto-init if we are on the login page
if (document.getElementById('login-form')) {
  Login.init();
}
