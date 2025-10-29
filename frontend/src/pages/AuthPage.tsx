import { useState } from "react";
import { useNavigate } from "react-router-dom";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;


export default function AuthPage() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  const isValidEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleSignUp = async () => {
    setErrorMessage("");

    if (!username || !email || !password) {
      setErrorMessage("Please fill all fields.");
      return;
    }

    if (!isValidEmail(email)) {
      setErrorMessage("Please enter a valid email.");
      return;
    }

    if (password.length < 6) {
      setErrorMessage("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8081/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });
      
      const data = await res.json();

      if (!res.ok) {
        setErrorMessage(data.message || "Signup failed");
      } else {
        alert("Account created successfully!");
        navigate("/"); // redirect after successful signup
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("Server error, please try again later.");
    }
    setLoading(false);
  };

  const handleLogin = async () => {
    setErrorMessage("");

    if (!email || !password) {
      setErrorMessage("Please enter both email and password.");
      return;
    }

    if (!isValidEmail(email)) {
      setErrorMessage("Please enter a valid email.");
      return;
    }

    setLoading(true);
    try {
     const res = await fetch(`${API_BASE_URL}/auth/signup`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username, email, password }),
});

      const data = await res.json();

      if (!res.ok) {
        setErrorMessage(data.message || "Login failed");
      } else {
        alert("Login successful!");
        navigate("/"); // redirect after successful login
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("Server error, please try again later.");
    }
    setLoading(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-black p-6">
      <div className="w-full max-w-md bg-gray-900 rounded-3xl shadow-2xl p-8 space-y-6 text-gray-100">
        {/* Back Button */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center text-gray-400 hover:text-white font-medium text-sm transition-colors duration-200"
        >
          ← Back
        </button>

        <h1 className="text-3xl font-bold text-white text-center">Welcome</h1>
        <p className="text-gray-400 text-center">Login or sign up to continue</p>

        {/* Form Inputs */}
        <div className="space-y-4 relative">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full px-4 py-3 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition bg-gray-800 text-white placeholder-gray-400"
          />

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-3 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition bg-gray-800 text-white placeholder-gray-400"
          />

          <div className="relative">
            <input
              type={showPassword ? "text" : "password"}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition bg-gray-800 text-white placeholder-gray-400"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-3 text-gray-400 hover:text-white text-sm"
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
        </div>

        {/* Error Message */}
        {errorMessage && (
          <p className="text-red-500 text-sm text-center">{errorMessage}</p>
        )}

        {/* Buttons */}
        <div className="flex flex-col space-y-3">
          <button
            onClick={handleLogin}
            disabled={loading}
            className="w-full py-3 bg-purple-600 text-white font-semibold rounded-xl hover:bg-purple-700 transition flex justify-center items-center"
          >
            {loading ? "Loading..." : "Login"}
          </button>

          <button
            onClick={handleSignUp}
            disabled={loading}
            className="w-full py-3 bg-teal-600 text-white font-semibold rounded-xl hover:bg-teal-700 transition flex justify-center items-center"
          >
            {loading ? "Loading..." : "Sign Up"}
          </button>
        </div>
      </div>
    </div>
  );
}
