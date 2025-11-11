import { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import L, { Icon } from "leaflet";
import axios from "axios";
import "leaflet/dist/leaflet.css";

// Extend react-leaflet typings
declare module "react-leaflet" {
  interface TileLayerProps {
    attribution?: string;
  }
}

const markerIcon: Icon = new L.Icon({
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

// ✅ Address shortener function
function formatAddress(fullAddress: string, addr: any): string {
  const importantParts = [
    addr.university || addr.building || addr.amenity || "",
    addr.suburb || addr.city_district || addr.town || "",
    addr.county || addr.city || "",
  ].filter(Boolean);

  let result = importantParts.join(", ");
  if (addr.postcode) result += ` - ${addr.postcode}`;

  // Fallback to top 3 parts if structure not matched
  if (result.split(",").length < 2 && fullAddress) {
    result = fullAddress.split(",").slice(0, 3).join(", ");
  }

  return result.trim();
}

interface MapPickerProps {
  onSelect: (data: {
    lat: number;
    lng: number;
    address: string;
    city?: string;
    state?: string;
  }) => void;
}

const LocationMarker = ({ onSelect }: MapPickerProps) => {
  const [position, setPosition] = useState<[number, number] | null>(null);

  const fetchAddress = async (lat: number, lng: number) => {
    try {
      const res = await axios.get(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
      );

      const addr = res.data.address || {};
      const rawAddress = res.data.display_name || "";
      const shortAddress = formatAddress(rawAddress, addr);

      onSelect({
        lat,
        lng,
        address: shortAddress,
        city: addr.city || addr.town || addr.village || "",
        state: addr.state || "",
      });
    } catch (err) {
      console.error("Reverse geocoding failed:", err);
    }
  };

  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      setPosition([lat, lng]);
      fetchAddress(lat, lng);
    },
  });

  return position ? <Marker position={position} {...({ icon: markerIcon } as any)} /> : null;
};

export const MapPicker = ({ onSelect }: MapPickerProps) => {
  const [center, setCenter] = useState<[number, number]>([20.5937, 78.9629]); // Default India

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setCenter([pos.coords.latitude, pos.coords.longitude]);
        },
        () => {
          console.warn("Geolocation permission denied. Using default center.");
        }
      );
    }
  }, []);

  return (
    <div className="h-[400px] w-full rounded-xl overflow-hidden relative">
      <MapContainer
        {...({
          center,
          zoom: 6,
          style: { height: "100%", width: "100%" },
          className: "z-0",
        } as any)}
      >
        <TileLayer
          attribution='&copy; <a href="https://osm.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <LocationMarker onSelect={onSelect} />
      </MapContainer>

      {/* Overlay instruction */}
      <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-black/60 text-white text-sm px-4 py-1.5 rounded-full shadow-md backdrop-blur-sm">
        🗺️ Click on the map to mark your issue location
      </div>
    </div>
  );
};
