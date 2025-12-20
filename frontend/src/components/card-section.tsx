"use client";

import * as React from "react";
import { IconTrendingDown, IconTrendingUp } from "@tabler/icons-react";
import { Spinner } from "@/components/ui/spinner";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function SectionCards() {
  const [currentPrice, setCurrentPrice] = React.useState<number | null>(null);
  const [priceChange24h, setPriceChange24h] = React.useState<number | null>(
    null
  );
  const [high24h, setHigh24h] = React.useState<number | null>(null);
  const [low24h, setLow24h] = React.useState<number | null>(null);
  const [tomorrowPrediction, setTomorrowPrediction] = React.useState<
    number | null
  >(null);
  const [loading, setLoading] = React.useState(true);
  const [predictionLoading, setPredictionLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    if (!mounted) return;
    const fetchPriceData = async () => {
      try {
        setError(null);

        const response = await fetch(
          "http://localhost:8000/api/predictions/current-price"
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        let priceValue: number | null = null;
        let changeValue: number | null = null;

        if (data.current_price) {
          if (typeof data.current_price === "number") {
            priceValue = data.current_price;
          } else if (
            typeof data.current_price === "object" &&
            data.current_price.current_price
          ) {
            priceValue = data.current_price.current_price;
          } else if (typeof data.current_price === "string") {
            // Structure 3: { current_price: "2973.7" }
            priceValue = parseFloat(data.current_price);
          }
        }

        if (data.price_change_24h) {
          if (typeof data.price_change_24h === "number") {
            changeValue = data.price_change_24h;
          } else if (
            typeof data.current_price === "object" &&
            data.current_price.price_change_24h
          ) {
            changeValue = data.current_price.price_change_24h;
          } else if (typeof data.price_change_24h === "string") {
            changeValue = parseFloat(data.price_change_24h);
          }
        }

        // Validate and set
        if (
          typeof priceValue === "number" &&
          !isNaN(priceValue) &&
          priceValue > 0
        ) {
          setCurrentPrice(priceValue);
        } else {
          setError("Invalid price data received");
        }

        if (typeof changeValue === "number" && !isNaN(changeValue)) {
          setPriceChange24h(changeValue);
        }

        // Fetch historical data for high/low
        const histResponse = await fetch(
          "http://localhost:8000/api/predictions/historical-prices?days=1&interval=hourly"
        );

        if (histResponse.ok) {
          const histData = await histResponse.json();
          const prices = histData.data.map((item: any) => {
            const price =
              typeof item.price === "number"
                ? item.price
                : parseFloat(item.price);
            return price;
          });

          if (prices.length > 0 && priceValue) {
            const allPrices = [...prices, priceValue];
            const high = Math.max(...allPrices);
            const low = Math.min(...allPrices);

            setHigh24h(high);
            setLow24h(low);
          }
        }
      } catch (error) {
        setError(
          error instanceof Error ? error.message : "Failed to fetch data"
        );
      } finally {
        setLoading(false);
      }
    };

    fetchPriceData();
    const interval = setInterval(fetchPriceData, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [mounted]);

  // Fetch Tomorrow Prediction
  React.useEffect(() => {
    if (!mounted) return;
    const fetchPrediction = async () => {
      try {
        console.log("🔮 Fetching prediction...");
        const response = await fetch(
          "http://localhost:8000/api/predictions/predict-tomorrow" // ✅ Fix: Ganti ke predict-tomorrow
        );

        console.log("📡 Response status:", response.status);

        if (response.ok) {
          const data = await response.json();
          console.log(
            "📊 Full Prediction Data:",
            JSON.stringify(data, null, 2)
          );

          // ✅ Handle response structure
          let prediction = null;

          if (data.tomorrow_predictions) {
            prediction =
              data.tomorrow_predictions.bayesian_model ||
              data.tomorrow_predictions.recommended ||
              data.tomorrow_predictions.basic_model;
            console.log("✅ Found prediction:", prediction);
          }

          if (
            typeof prediction === "number" &&
            !isNaN(prediction) &&
            prediction > 0
          ) {
            setTomorrowPrediction(prediction);
            console.log("✅ Tomorrow prediction set:", prediction);
          } else {
            console.warn("⚠️ Invalid prediction value:", prediction);
            setTomorrowPrediction(null);
          }
        } else {
          const errorData = await response.json();
          console.error("❌ Prediction API error:", errorData);
          setTomorrowPrediction(null);
        }
      } catch (error) {
        console.error("❌ Error fetching prediction:", error);
        setTomorrowPrediction(null);
      } finally {
        setPredictionLoading(false);
      }
    };

    fetchPrediction();
    // Refresh every 30 minutes
    const interval = setInterval(fetchPrediction, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, [mounted]);

  const formatPrice = (price: number | null) => {
    // ✅ Add safety check
    if (price === null || price === undefined || typeof price !== "number") {
      console.warn("Invalid price:", price);
      return "N/A";
    }

    return `$${price.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  const formatChange = (change: number | null) => {
    if (change === null || change === undefined) return "N/A";
    return `${change > 0 ? "+" : ""}${change.toFixed(2)}%`;
  };

  return (
    <div className="grid grid-cols-1 gap-4 px-4 lg:px-6 md:grid-cols-2 lg:grid-cols-4">
      {/* Current Price */}
      <Card>
        <CardHeader>
          <CardDescription>Current Price</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {loading ? (
              <Spinner className="size-8" />
            ) : currentPrice ? (
              formatPrice(currentPrice)
            ) : (
              "N/A"
            )}
          </CardTitle>
          <CardAction>
            {priceChange24h !== null ? (
              <Badge variant="outline">
                {priceChange24h >= 0 ? (
                  <IconTrendingUp className="h-4 w-4" />
                ) : (
                  <IconTrendingDown className="h-4 w-4" />
                )}
                {formatChange(priceChange24h)}
              </Badge>
            ) : (
              <Badge variant="outline">N/A</Badge>
            )}
          </CardAction>
        </CardHeader>
      </Card>

      {/* 24h High */}
      <Card>
        <CardHeader>
          <CardDescription>24h High</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {loading ? (
              <Spinner className="size-8" />
            ) : high24h ? (
              formatPrice(high24h)
            ) : (
              "N/A"
            )}
          </CardTitle>
        </CardHeader>
      </Card>

      {/* 24h Low */}
      <Card>
        <CardHeader>
          <CardDescription>24h Low</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {loading ? (
              <Spinner className="size-8" />
            ) : low24h ? (
              formatPrice(low24h)
            ) : (
              "N/A"
            )}
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Tomorrow Prediction */}
      <Card>
        <CardHeader>
          <CardDescription>Tomorrow Prediction</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {predictionLoading ? (
              <Spinner className="size-8" />
            ) : tomorrowPrediction ? (
              formatPrice(tomorrowPrediction)
            ) : (
              <span className="text-base text-muted-foreground">
                Train Model First
              </span>
            )}
          </CardTitle>
        </CardHeader>
      </Card>
    </div>
  );
}
