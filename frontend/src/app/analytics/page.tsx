"use client";

import * as React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { BarChart3, TrendingUp } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";

export default function AnalyticsPage() {
  const [loading, setLoading] = React.useState(true);
  const [featureImportance, setFeatureImportance] = React.useState<any>(null);
  const [modelComparison, setModelComparison] = React.useState<any>(null);

  React.useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        // Fetch feature importance
        const featResponse = await fetch(
          "http://localhost:8000/api/predictions/feature-importance"
        );
        if (featResponse.ok) {
          const featData = await featResponse.json();
          setFeatureImportance(featData.features);
        }

        // Fetch model comparison
        const statusResponse = await fetch(
          "http://localhost:8000/api/predictions/status"
        );
        if (statusResponse.ok) {
          const data = await statusResponse.json();
          if (data.training_results) {
            setModelComparison({
              basic: data.training_results.basic?.test_metrics,
              bayesian: data.training_results.bayesian?.test_metrics,
              basicTime: data.training_results.basic?.training_time,
              bayesianTime: data.training_results.bayesian?.training_time,
            });
          }
        }
      } catch (error) {
        console.error("Error fetching analytics:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator
              orientation="vertical"
              className="mr-2 data-[orientation=vertical]:h-4"
            />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="#">ETH Prediction</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbPage>Analytics</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 pt-4">
          <div className="px-4 lg:px-6 space-y-6">
            {/* Feature Importance */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  <CardTitle>Feature Importance</CardTitle>
                </div>
                <CardDescription>What drives the predictions?</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center py-8">
                    <Spinner className="h-8 w-8" />
                  </div>
                ) : featureImportance ? (
                  <div className="space-y-3">
                    {featureImportance.map((item: any, index: number) => (
                      <div key={index} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium">{item.feature}</span>
                          <span className="text-muted-foreground">
                            {item.importance.toFixed(1)}%
                          </span>
                        </div>
                        <Progress value={item.importance} className="h-2" />
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground py-8 text-center">
                    No data available. Please train models first.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Model Performance Comparison */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  <CardTitle>Model Performance Comparison</CardTitle>
                </div>
                <CardDescription>Which model works better?</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center py-8">
                    <Spinner className="h-8 w-8" />
                  </div>
                ) : modelComparison ? (
                  <div className="rounded-lg border">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b bg-muted/50">
                          <th className="p-3 text-left font-medium">Metric</th>
                          <th className="p-3 text-center font-medium">
                            Basic Model
                          </th>
                          <th className="p-3 text-center font-medium">
                            Bayesian Model
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b">
                          <td className="p-3 font-medium">RMSE</td>
                          <td className="p-3 text-center">
                            ${modelComparison.basic?.rmse?.toFixed(2) || "N/A"}
                          </td>
                          <td className="p-3 text-center">
                            $
                            {modelComparison.bayesian?.rmse?.toFixed(2) ||
                              "N/A"}
                          </td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-3 font-medium">MAE</td>
                          <td className="p-3 text-center">
                            ${modelComparison.basic?.mae?.toFixed(2) || "N/A"}
                          </td>
                          <td className="p-3 text-center">
                            $
                            {modelComparison.bayesian?.mae?.toFixed(2) || "N/A"}
                          </td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-3 font-medium">R² Score</td>
                          <td className="p-3 text-center">
                            {modelComparison.basic?.r2?.toFixed(4) || "N/A"}
                          </td>
                          <td className="p-3 text-center">
                            {modelComparison.bayesian?.r2?.toFixed(4) || "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-3 font-medium">Training Time</td>
                          <td className="p-3 text-center">
                            {modelComparison.basicTime?.toFixed(2) || "N/A"}s
                          </td>
                          <td className="p-3 text-center">
                            {modelComparison.bayesianTime?.toFixed(2) || "N/A"}s
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground py-8 text-center">
                    No data available. Please train models first.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
