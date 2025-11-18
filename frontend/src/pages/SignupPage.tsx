import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

const API_BASE_URL = import.meta.env.VITE_API_URL;

export default function SignupPage() {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    dob: "",
    email: "",
    username: "",
    password: "",
    confirmPassword: "",
  });

  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSignup = async () => {
    const { firstName, lastName, dob, email, username, password, confirmPassword } =
      formData;

    setErrorMessage("");

    if (!firstName || !lastName || !dob || !email || !username || !password || !confirmPassword) {
      setErrorMessage("Please fill all fields.");
      return;
    }

    if (password !== confirmPassword) {
      setErrorMessage("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrorMessage(data.message || "Signup failed");
      } else {
        navigate("/login");
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("Server error, please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen 
      bg-[#030712] p-6 text-blue-100">

      {/* Card */}
      <div className="relative w-full max-w-lg 
        bg-[#0b0f16]/90 backdrop-blur-xl rounded-3xl 
        shadow-[0_0_25px_rgba(0,102,255,0.3)]
        border border-blue-700/30 
        p-8 space-y-8">

        {/* Exit Button */}
        <button
          onClick={() => navigate("/")}
          className="absolute top-4 right-4 text-blue-400 hover:text-blue-300 
          text-3xl font-bold transition"
        >
          ×
        </button>

        {/* Title */}
        <h1 className="text-3xl font-bold text-center text-blue-300">
          Create Account
        </h1>
        <p className="text-blue-400/80 text-center">
          Fill the details below to get started
        </p>

        {/* Form Fields */}
        <div className="grid grid-cols-2 gap-4">

          <input
            name="firstName"
            type="text"
            placeholder="First Name"
            value={formData.firstName}
            onChange={handleChange}
            className="px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl placeholder-blue-400/60 
            focus:outline-none focus:ring-2 focus:ring-blue-500/70"
          />

          <input
            name="lastName"
            type="text"
            placeholder="Last Name"
            value={formData.lastName}
            onChange={handleChange}
            className="px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl placeholder-blue-400/60 
            focus:outline-none focus:ring-2 focus:ring-blue-500/70"
          />

          <input
            name="dob"
            type="date"
            value={formData.dob}
            onChange={handleChange}
            className="col-span-2 px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl focus:outline-none 
            focus:ring-2 focus:ring-blue-500/70"
          />

          <input
            name="email"
            type="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            className="col-span-2 px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl placeholder-blue-400/60 
            focus:outline-none focus:ring-2 focus:ring-blue-500/70"
          />

          <input
            name="username"
            type="text"
            placeholder="Username"
            value={formData.username}
            onChange={handleChange}
            className="col-span-2 px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl placeholder-blue-400/60 
            focus:outline-none focus:ring-2 focus:ring-blue-500/70"
          />

          <input
            name="password"
            type="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            className="px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl placeholder-blue-400/60 
            focus:outline-none focus:ring-2 focus:ring-blue-500/70"
          />

          <input
            name="confirmPassword"
            type="password"
            placeholder="Confirm Password"
            value={formData.confirmPassword}
            onChange={handleChange}
            className="px-4 py-3 bg-[#111827] border border-blue-600/30 
            text-blue-200 rounded-xl placeholder-blue-400/60 
            focus:outline-none focus:ring-2 focus:ring-blue-500/70"
          />
        </div>

        {/* Error */}
        {errorMessage && (
          <p className="text-red-400 text-sm text-center">{errorMessage}</p>
        )}

        {/* Submit Button */}
        <button
          onClick={handleSignup}
          disabled={loading}
          className="w-full py-3 rounded-xl font-semibold text-white
          bg-blue-600 hover:bg-blue-700 
          shadow-[0_0_15px_rgba(0,102,255,0.5)]
          hover:shadow-[0_0_20px_rgba(0,102,255,0.7)]
          transition-all duration-300"
        >
          {loading ? "Creating account..." : "Sign Up"}
        </button>

        {/* Login Link */}
        <p className="text-blue-400/80 text-center text-sm">
          Already have an account?{" "}
          <Link
            to="/login"
            className="text-blue-300 hover:underline hover:text-blue-200"
          >
            Log in
          </Link>
        </p>

      </div>
    </div>
  );
}
