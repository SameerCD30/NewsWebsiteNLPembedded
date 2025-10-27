import { MapPin } from "lucide-react";
import { Button } from "./ui/button";
import { useNavigate } from "react-router-dom";

export const Header = () => {
  const navigate = useNavigate();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16 bg-gradient-to-r from-[#1a1a1d]/95 to-[#2a2a2d]/95 backdrop-blur-xl border-b border-border/50 shadow-lg">
      <div className="flex h-full items-center justify-between px-8">
        
        {/* LOGO */}
        <h1
  onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
  className="text-3xl font-extrabold tracking-tight 
             bg-gradient-to-r from-rose-500 via-red-500 to-orange-400 
             bg-clip-text text-transparent 
             drop-shadow-[0_2px_8px_rgba(255,100,100,0.6)] 
             cursor-pointer transition-all duration-300 
             hover:scale-105 hover:drop-shadow-[0_4px_12px_rgba(255,100,100,0.8)]"
>
  Grievance Beacon
</h1>


        {/* RIGHT SECTION */}
        <div className="flex items-center gap-6">
          {/* Change Location */}
          <button
            className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-primary transition-all duration-300 hover:scale-105"
            onClick={() =>
              window.open("https://maps.google.com", "_blank")
            }
          >
            <MapPin className="h-4 w-4 text-primary" />
            Change Location
          </button>

          {/* Login / Signup */}
          <Button
  onClick={() => navigate("/auth")}
  className="bg-gradient-to-r from-red-500 to-rose-600 text-white font-semibold px-6 py-2 rounded-lg 
             shadow-[0_4px_10px_rgba(255,0,0,0.4)] 
             transition-all duration-300 
             hover:from-red-600 hover:to-rose-700 
             hover:shadow-[0_6px_14px_rgba(255,0,0,0.6)] 
             hover:scale-105 active:scale-95"
>
  Login / SignUp
</Button>

        </div>
      </div>
    </header>
  );
};
