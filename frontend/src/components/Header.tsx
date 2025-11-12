import { MapPin } from "lucide-react";
import { Button } from "./ui/button";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";

interface HeaderProps {
  onChangeLocation: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onChangeLocation }) => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

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
            type="button"
            onClick={() => {
              console.log("Change Location clicked!");
              onChangeLocation();
            }}
            className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-primary transition-all duration-300 hover:scale-105"
          >
            <MapPin className="h-4 w-4 text-primary pointer-events-none" />
            Change Location
          </button>

          {/* Auth Section */}
          {!user ? (
            <Button
              onClick={() => navigate("/login")}
              className="bg-gradient-to-r from-red-500 to-rose-600 text-white font-semibold px-6 py-2 rounded-lg 
                     shadow-[0_4px_10px_rgba(255,0,0,0.4)] 
                     transition-all duration-300 
                     hover:from-red-600 hover:to-rose-700 
                     hover:shadow-[0_6px_14px_rgba(255,0,0,0.6)] 
                     hover:scale-105 active:scale-95"
            >
              Login / Sign Up
            </Button>
          ) : (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="rounded-full border-2 border-red-600 hover:scale-105 transition">
                  <Avatar className="h-10 w-10">
                    <AvatarImage
                      src={
                        user.profilePic ||
                        "https://cdn-icons-png.flaticon.com/512/149/149071.png"
                      }
                      alt={user.username}
                    />
                    <AvatarFallback>
                      {user.username ? user.username[0].toUpperCase() : "U"}
                    </AvatarFallback>
                  </Avatar>
                </button>
              </DropdownMenuTrigger>

              <DropdownMenuContent
                align="end"
                className="w-44 bg-[#1b1b1b] border border-gray-700 text-gray-200 rounded-lg shadow-lg"
              >
                <DropdownMenuItem
                  onClick={() => navigate("/myposts")}
                  className="cursor-pointer hover:bg-red-600/20 transition"
                >
                  🧾 My Posts
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={logout}
                  className="cursor-pointer text-red-400 hover:bg-red-600/20 transition"
                >
                  🚪 Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
