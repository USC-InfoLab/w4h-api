/**
 * HIPAA-compliant password validation
 * Requirements: 12+ characters with uppercase, lowercase, number, and special character
 */
function validatePassword(password) {
  if (!password) {
    return {
      valid: false,
      message: 'Password is required'
    };
  }
  const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#^()_+=\-[\]{}|\\:;"'<>,./`~]).{12,}$/;
  if (!regex.test(password)) {
    return {
      valid: false,
      message: 'Password must be at least 12 characters and include uppercase, lowercase, number, and special character'
    };
  }
  return { valid: true };
}

export { validatePassword };
