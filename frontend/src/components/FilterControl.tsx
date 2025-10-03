import { useState } from "react";
import { cn } from "@/lib/utils";

type FilterLevel = "Local" | "State" | "National";

export const FilterControl = () => {
  const [activeFilter, setActiveFilter] = useState<FilterLevel>("Local");

  const filters: FilterLevel[] = ["Local", "State", "National"];

  return (
    <div className="flex justify-center mb-10">
      <div className="inline-flex rounded-full bg-card/80 border border-border/60 p-1.5 shadow-xl backdrop-blur-sm">
        {filters.map((filter) => (
          <button
            key={filter}
            onClick={() => setActiveFilter(filter)}
            className={cn(
              "px-8 py-2.5 rounded-full text-sm font-semibold transition-all duration-300 relative",
              activeFilter === filter
                ? "bg-primary text-primary-foreground shadow-lg scale-105"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            {filter}
          </button>
        ))}
      </div>
    </div>
  );
};
