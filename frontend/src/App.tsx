import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Feed from "@/components/Feed"; 
import NotFound from "./pages/NotFound";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import MyPosts from "./pages/MyPosts";
import { AuthProvider, useAuth } from "@/context/AuthContext";

const queryClient = new QueryClient();

const ProtectedRoute = ({ children }: { children: JSX.Element }) => {
  const { user } = useAuth();
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            {/*  Home now renders Feed */}
            <Route path="/" element={<Feed />} />

            {/* Auth routes */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />

            {/* Protected Route: My Posts */}
            <Route
              path="/myposts"
              element={
                <ProtectedRoute>
                  <MyPosts />
                </ProtectedRoute>
              }
            />

            {/* Catch-all */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
