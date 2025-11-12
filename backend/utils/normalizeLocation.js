// utils/normalizeLocation.js

/**
 * Normalizes a given state and city for consistent lookups.
 * Ensures lowercase keys and trims unwanted spaces.
 * Handles edge cases like "New Delhi" vs "delhi" etc.
 */

module.exports = function normalizeLocation(city = "", state = "") {
  if (!city || !state) {
    return { cityKey: "", stateKey: "" };
  }

  // Trim + lowercase everything
  const cleanCity = city.trim().toLowerCase();
  const cleanState = state.trim().toLowerCase();

  // Handle some known aliases for better matching
  const aliasMap = {
    "new delhi": "delhi",
    "ncr": "delhi",
    "gurgaon": "gurugram",
    "bangalore": "bengaluru",
    "mangalore": "mangaluru",
    "trivandrum": "thiruvananthapuram",
    "calcutta": "kolkata",
    "madras": "chennai",
  };

  const cityKey = aliasMap[cleanCity] || cleanCity;
  const stateKey = aliasMap[cleanState] || cleanState;

  return { cityKey, stateKey };
};
