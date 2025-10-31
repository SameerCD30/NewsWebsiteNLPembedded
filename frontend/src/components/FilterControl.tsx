import { useState } from "react";
import { cn } from "@/lib/utils";

type FilterLevel = "Local" | "State" | "National";

export const FilterControl = () => {
  const [activeFilter, setActiveFilter] = useState<FilterLevel>("Local");

  const filters: FilterLevel[] = ["Local", "State", "National"];

  return (
    <div className="flex justify-center mb-10">
      <div className="inline-flex rounded-full bg-zinc-900/60 border border-zinc-700 p-1.5 shadow-lg backdrop-blur-sm">
        {filters.map((filter) => (
          <button
            key={filter}
            onClick={() => setActiveFilter(filter)}
            className={cn(
              "relative px-8 py-2.5 rounded-full text-sm font-semibold transition-all duration-300 overflow-hidden",
              activeFilter === filter
                ? "text-white bg-gradient-to-r from-red-500 to-orange-400 shadow-[0_0_10px_rgba(255,87,34,0.5)] scale-105"
                : "text-zinc-300 hover:text-white hover:bg-zinc-800/70"
            )}
          >
            {filter}

            {/* Subtle glow animation on active button */}
            {activeFilter === filter && (
              <span className="absolute inset-0 rounded-full bg-gradient-to-r from-red-500 to-orange-400 opacity-30 blur-md animate-pulse -z-10" />
            )}
          </button>
        ))}
      </div>
    </div>
  );
};
