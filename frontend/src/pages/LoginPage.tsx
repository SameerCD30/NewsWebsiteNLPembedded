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
      alert("Login successful!");
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
    <div className="flex items-center justify-center min-h-screen bg-black p-6">
      <div className="relative w-full max-w-md bg-gray-900 rounded-3xl shadow-2xl p-8 space-y-6 text-gray-100">
        <button
          onClick={() => navigate("/")}
          className="absolute top-4 right-4 text-gray-400 hover:text-white text-2xl font-bold"
        >
          ×
        </button>

        <h1 className="text-3xl font-bold text-center text-white">
          Welcome Back
        </h1>
        <p className="text-gray-400 text-center">Login to your account</p>

        <div className="space-y-4">
          <input
            type="text"
            placeholder="Username or Email"
            value={identifier}
            onChange={(e) => setIdentifier(e.target.value)}
            className="w-full px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
        </div>

        {errorMessage && (
          <p className="text-red-500 text-sm text-center">{errorMessage}</p>
        )}

        <button
          onClick={handleLogin}
          disabled={loading}
          className="w-full py-3 bg-purple-600 rounded-xl hover:bg-purple-700 transition font-semibold"
        >
          {loading ? "Logging in..." : "Login"}
        </button>

        <p className="text-gray-400 text-center text-sm">
          Don’t have an account?{" "}
          <Link to="/signup" className="text-teal-400 hover:underline">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}
