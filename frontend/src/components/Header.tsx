import { MapPin } from "lucide-react";
import { Button } from "./ui/button";

export const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-border/50 bg-card/95 backdrop-blur-xl supports-[backdrop-filter]:bg-card/80 shadow-lg">
      <div className="flex h-full items-center justify-between px-8">
        <h1 className="text-2xl font-bold tracking-tight text-danger drop-shadow-sm">
          Grievance Beacon
        </h1>
        
        <div className="flex items-center gap-6">
          <button className="flex items-center gap-2 text-sm font-medium text-foreground hover:text-primary transition-all duration-200 hover:scale-105">
            <MapPin className="h-4 w-4" />
            Change Location
          </button>
          
          <Button variant="danger" size="default" className="shadow-md hover:shadow-lg transition-all duration-200">
            Login / SignUp
          </Button>
        </div>
      </div>
    </header>
  );
};
