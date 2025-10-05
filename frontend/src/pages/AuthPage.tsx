import { useState } from "react";
import { supabase } from "../supabaseClient";
import { useNavigate } from "react-router-dom";

export default function AuthPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  const isValidEmail = (email: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleSignUp = async () => {
    setErrorMessage("");
    if (!email || !password) {
      setErrorMessage("Please enter both email and password.");
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
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) setErrorMessage(error.message);
    else alert("Check your email for confirmation link!");
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
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) setErrorMessage(error.message);
    else navigate("/"); 
    setLoading(false);
  };

  const handleGoogleLogin = async () => {
    const { error } = await supabase.auth.signInWithOAuth({ provider: "google" });
    if (error) setErrorMessage(error.message);
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
          <button
            onClick={handleGoogleLogin}
            className="w-full py-3 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" viewBox="0 0 533.5 544.3">
              <path
                fill="#fff"
                d="M533.5 278.4c0-18.8-1.5-37-4.4-54.7H272v103.7h146.9c-6.3 34.1-25 63-53.4 82.4v68h86.4c50.6-46.6 79.6-115.4 79.6-199.4z"
              />
              <path
                fill="#fff"
                d="M272 544.3c72.5 0 133.2-23.9 177.6-64.7l-86.4-68c-24 16.2-55 25.7-91.2 25.7-69.9 0-129.3-47.2-150.4-110.6H33.1v69.7C77.5 491 169 544.3 272 544.3z"
              />
              <path
                fill="#fff"
                d="M121.6 337.4c-4.7-13.9-7.4-28.6-7.4-43.4s2.7-29.5 7.4-43.4V180.6H33.1c-14.3 28.5-22.5 60.7-22.5 95s8.2 66.5 22.5 95l88.5-69.2z"
              />
              <path
                fill="#fff"
                d="M272 108.3c39.5 0 75 13.6 102.9 40.4l77.2-77.2C405 24 344.3 0 272 0 169 0 77.5 53.3 33.1 140.3l88.5 69.7C142.7 155.5 202.1 108.3 272 108.3z"
              />
            </svg>
            Sign in with Google
          </button>
        </div>
      </div>
    </div>
  );
}
