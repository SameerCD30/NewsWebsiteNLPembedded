import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { loginUser } from "../api/api";
import { useAuth } from "@/context/AuthContext";

export default function LoginPage() {
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLogin = async () => {
    setErrorMessage("");

    if (!identifier || !password) {
      setErrorMessage("Please enter both username/email and password.");
      return;
    }

    setLoading(true);
    try {
      const response = await loginUser({ email: identifier, password });
      const { token, user } = response.data;

      if (!token || !user) throw new Error("Invalid login response.");

      login(user, token);
      navigate("/");
    } catch (err: any) {
      console.error(err);
      const msg =
        err?.response?.data?.message ||
        err?.message ||
        "Invalid credentials or server error.";
      setErrorMessage(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen 
      bg-[#030712] text-blue-100 p-6">

      {/* Card */}
      <div className="relative w-full max-w-md 
        bg-[#0b0f16]/90 backdrop-blur-xl rounded-3xl 
        border border-blue-700/30 
        shadow-[0_0_25px_rgba(0,102,255,0.25)]
        p-8 space-y-8">

        {/* Close button */}
        <button
          onClick={() => navigate("/")}
          className="absolute top-4 right-4 text-blue-400 hover:text-blue-300 
          text-3xl font-bold transition"
        >
          ×
        </button>

        {/* Title */}
        <h1 className="text-3xl font-bold text-center text-blue-300">
          Welcome Back
        </h1>
        <p className="text-blue-400/80 text-center">
          Login to your account to continue
        </p>

        {/* Inputs */}
        <div className="space-y-5">
          <div>
            <input
              type="text"
              placeholder="Username or Email"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              className="w-full px-4 py-3 rounded-xl 
              bg-[#111827] border border-blue-700/30 text-blue-200
              placeholder-blue-400/50
              focus:outline-none focus:ring-2 focus:ring-blue-500/70
              transition"
            />
          </div>

          <div>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 rounded-xl 
              bg-[#111827] border border-blue-700/30 text-blue-200
              placeholder-blue-400/50
              focus:outline-none focus:ring-2 focus:ring-blue-500/70
              transition"
            />
          </div>
        </div>

        {/* Error Message */}
        {errorMessage && (
          <p className="text-red-400 text-sm text-center font-medium">
            {errorMessage}
          </p>
        )}

        {/* Login Button */}
        <button
          onClick={handleLogin}
          disabled={loading}
          className="w-full py-3 rounded-xl font-semibold text-white
          bg-blue-600 hover:bg-blue-700 
          shadow-[0_0_15px_rgba(0,102,255,0.6)]
          hover:shadow-[0_0_20px_rgba(0,102,255,0.8)]
          transition-all duration-300"
        >
          {loading ? "Logging in..." : "Login"}
        </button>

        {/* Signup Link */}
        <p className="text-blue-400/80 text-center text-sm">
          Don’t have an account?{" "}
          <Link
            to="/signup"
            className="text-blue-300 hover:underline hover:text-blue-200"
          >
            Create one
          </Link>
        </p>

      </div>
    </div>
  );
}
