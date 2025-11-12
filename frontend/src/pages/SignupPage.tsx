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
    const { firstName, lastName, dob, email, username, password, confirmPassword } = formData;
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
        alert("Account created successfully!");
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
    <div className="flex items-center justify-center min-h-screen bg-black p-6">
      <div className="relative w-full max-w-lg bg-gray-900 rounded-3xl shadow-2xl p-8 space-y-6 text-gray-100">
        <button
          onClick={() => navigate("/")}
          className="absolute top-4 right-4 text-gray-400 hover:text-white text-2xl font-bold"
        >
          ×
        </button>

        <h1 className="text-3xl font-bold text-center text-white">Create Account</h1>
        <p className="text-gray-400 text-center">Fill the details below</p>

        <div className="grid grid-cols-2 gap-4">
          <input
            name="firstName"
            type="text"
            placeholder="First Name"
            value={formData.firstName}
            onChange={handleChange}
            className="col-span-1 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <input
            name="lastName"
            type="text"
            placeholder="Last Name"
            value={formData.lastName}
            onChange={handleChange}
            className="col-span-1 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <input
            name="dob"
            type="date"
            value={formData.dob}
            onChange={handleChange}
            className="col-span-2 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <input
            name="email"
            type="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            className="col-span-2 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <input
            name="username"
            type="text"
            placeholder="Username"
            value={formData.username}
            onChange={handleChange}
            className="col-span-2 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <input
            name="password"
            type="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            className="col-span-1 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <input
            name="confirmPassword"
            type="password"
            placeholder="Confirm Password"
            value={formData.confirmPassword}
            onChange={handleChange}
            className="col-span-1 px-4 py-3 border border-gray-700 rounded-xl bg-gray-800 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
        </div>

        {errorMessage && (
          <p className="text-red-500 text-sm text-center">{errorMessage}</p>
        )}

        <button
          onClick={handleSignup}
          disabled={loading}
          className="w-full py-3 bg-teal-600 rounded-xl hover:bg-teal-700 transition font-semibold"
        >
          {loading ? "Creating account..." : "Sign Up"}
        </button>

        <p className="text-gray-400 text-center text-sm">
          Already have an account?{" "}
          <Link to="/login" className="text-purple-400 hover:underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
}
