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
    <header className="fixed top-0 left-0 right-0 z-50 h-16 
      bg-[#0b0f16]/80 backdrop-blur-2xl 
      border-b border-blue-600/20 
      shadow-[0_0_15px_rgba(0,102,255,0.25)]">
      <div className="flex h-full items-center justify-between px-8">

        {/* LOGO */}
        <h1
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          className="text-3xl font-extrabold tracking-tight cursor-pointer 
            bg-gradient-to-r from-blue-400 via-blue-500 to-cyan-300 
            bg-clip-text text-transparent 
            drop-shadow-[0_0_10px_rgba(0,122,255,0.7)]
            transition-all duration-300 
            hover:scale-105 hover:drop-shadow-[0_0_16px_rgba(0,122,255,0.9)]"
        >
          Grievance Beacon
        </h1>

        {/* RIGHT SIDE */}
        <div className="flex items-center gap-6">

          {/* Change Location */}
          <button
            type="button"
            onClick={() => onChangeLocation()}
            className="flex items-center gap-2 text-sm font-medium 
              text-gray-300 hover:text-blue-400 
              transition-all duration-300 hover:scale-105"
          >
            <MapPin className="h-4 w-4 text-blue-400 pointer-events-none" />
            Change Location
          </button>

          {/* Auth Buttons */}
          {!user ? (
            <Button
              onClick={() => navigate("/login")}
              className="bg-blue-600 text-white font-semibold px-6 py-2 rounded-lg 
                shadow-[0_0_12px_rgba(0,102,255,0.45)] 
                transition-all duration-300 
                hover:bg-blue-700 hover:shadow-[0_0_18px_rgba(0,102,255,0.65)] 
                hover:scale-105 active:scale-95"
            >
              Login / Sign Up
            </Button>
          ) : (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="rounded-full border-2 border-blue-500 
                  hover:scale-105 transition shadow-[0_0_8px_rgba(0,102,255,0.4)]">
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
                className="w-44 bg-[#0f141c] border border-blue-600/30 
                  text-gray-200 rounded-lg shadow-xl"
              >
                <DropdownMenuItem
                  onClick={() => navigate("/myposts")}
                  className="cursor-pointer hover:bg-blue-600/20 transition"
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
