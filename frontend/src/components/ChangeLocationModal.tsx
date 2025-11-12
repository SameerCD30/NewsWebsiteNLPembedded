import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Location } from "@/types/location";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import statesData from "@/data/stateCities";

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (newLocation: Location) => void;
  currentLocation: Location;
}

const ChangeLocationModal: React.FC<Props> = ({
  open,
  onClose,
  onSave,
  currentLocation,
}) => {
  const [selectedState, setSelectedState] = useState(currentLocation.state || "");
  const [selectedCity, setSelectedCity] = useState(currentLocation.city || "");
  const [country] = useState("India");

  useEffect(() => {
    setSelectedState(currentLocation.state || "");
    setSelectedCity(currentLocation.city || "");
  }, [currentLocation]);

  const handleSave = () => {
    if (!selectedState || !selectedCity) {
      alert("Please select both state and city");
      return;
    }

    const newLoc = { city: selectedCity, state: selectedState, country };
    onSave(newLoc);
    localStorage.setItem("userLocation", JSON.stringify(newLoc));
    onClose();
  };

  const allStates = Object.keys(statesData).sort((a, b) => a.localeCompare(b));

  // ✅ Fix: no lowercasing — directly access the state key
  const citiesForSelectedState =
    selectedState && statesData[selectedState]
      ? statesData[selectedState]
      : [];

  return (
    <AnimatePresence>
      {open && (
        <Dialog open={open} onOpenChange={onClose}>
          <DialogContent className="bg-card/95 backdrop-blur-xl border border-border/50 rounded-2xl shadow-lg sm:max-w-md">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.25 }}
            >
              <DialogHeader>
                <DialogTitle className="text-lg font-semibold text-foreground">
                  Change Location
                </DialogTitle>
              </DialogHeader>

              <div className="flex flex-col gap-4 mt-3">
                {/* STATE DROPDOWN */}
                <div>
                  <label className="text-sm text-muted-foreground">
                    Select State
                  </label>
                  <select
                    value={selectedState}
                    onChange={(e) => {
                      setSelectedState(e.target.value);
                      setSelectedCity("");
                    }}
                    className="w-full mt-1 p-2 rounded-md bg-background border border-border text-foreground"
                  >
                    <option value="">-- Select State --</option>
                    {allStates.map((state) => (
                      <option key={state} value={state}>
                        {state}
                      </option>
                    ))}
                  </select>
                </div>

                {/* CITY DROPDOWN */}
                <div>
                  <label className="text-sm text-muted-foreground">
                    Select City
                  </label>
                  <select
                    value={selectedCity}
                    onChange={(e) => setSelectedCity(e.target.value)}
                    disabled={!selectedState}
                    className="w-full mt-1 p-2 rounded-md bg-background border border-border text-foreground disabled:opacity-50"
                  >
                    <option value="">-- Select City --</option>
                    {citiesForSelectedState.length > 0 ? (
                      citiesForSelectedState.map((city) => (
                        <option key={city} value={city}>
                          {city}
                        </option>
                      ))
                    ) : (
                      <option disabled>No cities available for this state</option>
                    )}
                  </select>
                </div>
              </div>

              <DialogFooter className="flex justify-end mt-5 gap-3">
                <Button variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button onClick={handleSave}>Save</Button>
              </DialogFooter>
            </motion.div>
          </DialogContent>
        </Dialog>
      )}
    </AnimatePresence>
  );
};

export default ChangeLocationModal;
