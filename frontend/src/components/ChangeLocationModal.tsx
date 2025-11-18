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

  const citiesForSelectedState =
    selectedState && statesData[selectedState]
      ? statesData[selectedState]
      : [];

  return (
    <AnimatePresence>
      {open && (
        <Dialog open={open} onOpenChange={onClose}>
          <DialogContent
            className="bg-[#0b0f16]/95 backdrop-blur-2xl rounded-2xl 
            border border-blue-700/40 shadow-[0_0_20px_rgba(0,102,255,0.25)]
            text-blue-100 sm:max-w-md p-0 overflow-hidden"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.25 }}
              className="p-6"
            >
              <DialogHeader>
                <DialogTitle className="text-xl font-semibold text-blue-300">
                  Change Location
                </DialogTitle>
              </DialogHeader>

              <div className="flex flex-col gap-6 mt-4">

                {/* STATE SELECT */}
                <div>
                  <label className="text-sm text-blue-400">Select State</label>
                  <select
                    value={selectedState}
                    onChange={(e) => {
                      setSelectedState(e.target.value);
                      setSelectedCity("");
                    }}
                    className="w-full mt-2 p-3 rounded-xl
                    bg-[#111827] text-blue-200
                    border border-blue-600/40
                    focus:outline-none focus:ring-2 focus:ring-blue-500/70
                    transition"
                  >
                    <option value="">-- Select State --</option>
                    {allStates.map((state) => (
                      <option key={state} value={state}>
                        {state}
                      </option>
                    ))}
                  </select>
                </div>

                {/* CITY SELECT */}
                <div>
                  <label className="text-sm text-blue-400">Select City</label>
                  <select
                    value={selectedCity}
                    onChange={(e) => setSelectedCity(e.target.value)}
                    disabled={!selectedState}
                    className="w-full mt-2 p-3 rounded-xl
                    bg-[#111827] text-blue-200
                    border border-blue-600/40
                    disabled:opacity-50
                    focus:outline-none focus:ring-2 focus:ring-blue-500/70
                    transition"
                  >
                    <option value="">-- Select City --</option>
                    {citiesForSelectedState.length > 0 ? (
                      citiesForSelectedState.map((city) => (
                        <option key={city} value={city}>
                          {city}
                        </option>
                      ))
                    ) : (
                      <option disabled>No cities available</option>
                    )}
                  </select>
                </div>
              </div>

              {/* FOOTER */}
              <DialogFooter className="mt-8 flex justify-end gap-3">
                <Button
                  variant="outline"
                  onClick={onClose}
                  className="border-blue-600/40 text-blue-300 hover:text-blue-200"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSave}
                  className="bg-blue-600 hover:bg-blue-700 
                  shadow-[0_0_12px_rgba(0,102,255,0.5)]"
                >
                  Save
                </Button>
              </DialogFooter>
            </motion.div>
          </DialogContent>
        </Dialog>
      )}
    </AnimatePresence>
  );
};

export default ChangeLocationModal;
