"use client";

import * as React from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Types
interface PriceData {
  date: string;
  price: number;
}

const chartConfig = {
  price: {
    label: "ETH Price",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

export function EthPriceChart() {
  const [timeframe, setTimeframe] = React.useState("30");
  const [data, setData] = React.useState<PriceData[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Fetch historical data
  const fetchHistoricalData = React.useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Determine interval based on timeframe
      const interval = timeframe === "1" ? "hourly" : "daily";

      // API call to backend
      const response = await fetch(
        `http://localhost:8000/api/predictions/historical-prices?days=${timeframe}&interval=${interval}`
      );

      if (!response.ok) {
        throw new Error(
          `API returned ${response.status}: ${response.statusText}`
        );
      }

      const apiResponse = await response.json();

      // Validate response structure
      if (!apiResponse.data || !Array.isArray(apiResponse.data)) {
        throw new Error("Invalid response format from API");
      }

      // Transform data for chart
      const chartData = apiResponse.data.map((item: any) => ({
        date: item.date,
        price: item.price,
      }));

      setData(chartData);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to fetch price data";

      setError(errorMessage);
      setData([]);
    } finally {
      setLoading(false);
    }
  }, [timeframe]);

  // Format price untuk tooltip
  const formatPrice = (value: number) => {
    return `$${value.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  // Initial data fetch
  React.useEffect(() => {
    fetchHistoricalData();
  }, [fetchHistoricalData]);

  // Smart auto refresh - only when tab is active
  React.useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    let lastFetch = Date.now();

    const startInterval = () => {
      interval = setInterval(() => {
        if (!document.hidden) {
          fetchHistoricalData();
          lastFetch = Date.now();
        }
      }, 900000); // 15 minutes
    };

    const stopInterval = () => {
      if (interval) {
        clearInterval(interval);
        interval = null;
      }
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopInterval();
      } else {
        const timeSinceLastFetch = Date.now() - lastFetch;
        const fiveMinutes = 5 * 60 * 1000;

        // Only fetch if more than 5 minutes since last fetch
        if (timeSinceLastFetch > fiveMinutes) {
          fetchHistoricalData();
          lastFetch = Date.now();
        }

        startInterval();
      }
    };

    startInterval();
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      stopInterval();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [fetchHistoricalData]);

  return (
    <Card>
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-3 sm:flex-row">
        <div className="grid flex-1 gap-2 text-center sm:text-left">
          <CardTitle>Ethereum Price Chart</CardTitle>
          <CardDescription>
            {loading
              ? "Loading price data..."
              : `Historical price data ${timeframe} days`}
          </CardDescription>
        </div>
        <Select value={timeframe} onValueChange={setTimeframe}>
          <SelectTrigger className="w-40" disabled={loading}>
            <SelectValue placeholder="Select timeframe" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">24 Hours</SelectItem>
            <SelectItem value="7">7 Days</SelectItem>
            <SelectItem value="30">30 Days</SelectItem>
            <SelectItem value="90">90 Days</SelectItem>
            <SelectItem value="365">1 Year</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center h-75">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : (
          <ChartContainer
            config={chartConfig}
            className="aspect-auto h-75 w-full"
          >
            <LineChart
              accessibilityLayer
              data={data}
              margin={{
                left: 12,
                right: 12,
                top: 12,
                bottom: 12,
              }}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                tickFormatter={(value) => {
                  const date = new Date(value);

                  // Kalau timeframe 1 day (24 hours), tampilkan jam
                  if (timeframe === "1") {
                    return date.toLocaleTimeString("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false, // 24-hour format
                    });
                  }

                  // Untuk timeframe lain, tampilkan tanggal
                  return date.toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  });
                }}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
                domain={["dataMin - 200", "dataMax + 200"]}
                scale="linear"
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(value) => {
                      return new Date(value).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      });
                    }}
                    formatter={(value) => [
                      formatPrice(Number(value)),
                      "ETH Price",
                    ]}
                  />
                }
              />
              <Line
                dataKey="price"
                type="monotone"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
