"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Badge } from "@/components/ui/badge";
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
import {
  Loader2,
  Zap,
  CheckCircle,
  Play,
  Square,
  BarChart3,
  Database,
  FileText,
} from "lucide-react";
import { Spinner } from "@/components/ui/spinner";

export default function TrainingPage() {
  const [dataSource, setDataSource] = React.useState("realtime");
  const [days, setDays] = React.useState(90);
  const [isLoading, setIsLoading] = React.useState(false);
  const [isTraining, setIsTraining] = React.useState(false);
  const [dataLoaded, setDataLoaded] = React.useState(false);
  const [dataRows, setDataRows] = React.useState(0);
  const [model1Progress, setModel1Progress] = React.useState(0);
  const [model2Progress, setModel2Progress] = React.useState(0);
  const [trainingResult, setTrainingResult] = React.useState<any>(null);
  const [logs, setLogs] = React.useState<string[]>([]);
  const [showResults, setShowResults] = React.useState(false);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    setLogs((prev) => [`${timestamp} - ${message}`, ...prev].slice(0, 10));
  };

  const handleLoadData = async () => {
    setIsLoading(true);
    setDataLoaded(false);
    addLog("Loading data...");

    try {
      const response = await fetch(
        `http://localhost:8000/api/predictions/data-info?days=${days}`
      );

      if (response.ok) {
        const data = await response.json();
        setDataRows(data.total_rows);
        setDataLoaded(true);
        addLog(`Data loaded successfully: ${data.total_rows} rows`);
      } else {
        addLog("Failed to load data");
      }
    } catch (error) {
      addLog("Error loading data");
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartTraining = async () => {
    if (!dataLoaded) {
      addLog("Please load data first");
      return;
    }

    setIsTraining(true);
    setModel1Progress(0);
    setModel2Progress(0);
    setTrainingResult(null);
    setShowResults(false);
    addLog("Training started");

    try {
      // Simulate progress for BOTH models simultaneously
      const progressInterval = setInterval(() => {
        setModel1Progress((prev) => {
          if (prev >= 100) return 100;
          return prev + 10;
        });
        setModel2Progress((prev) => {
          if (prev >= 100) return 100;
          return prev + 8; // Slightly slower for Bayesian
        });
      }, 500);

      const response = await fetch(
        `http://localhost:8000/api/predictions/train?historical_days=${days}`,
        { method: "POST" }
      );

      clearInterval(progressInterval);
      setModel1Progress(100);
      setModel2Progress(100);

      const data = await response.json();

      setTrainingResult(data);

      if (data.status === "success") {
        addLog("Basic XGBoost model trained");
        addLog("Bayesian XGBoost model trained");
        addLog("Training completed successfully");
      } else {
        addLog("Training failed: " + (data.detail || data.message));
      }
    } catch (error) {
      addLog("Training error occurred");
      setTrainingResult({ status: "error", message: "Training failed" });
    } finally {
      setIsTraining(false);
    }
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    addLog("Training stopped by user");
  };

  const handleViewResults = () => {
    setShowResults(!showResults);
  };

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        {/* Header with Breadcrumbs */}
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
                  <BreadcrumbPage>Training</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-1 flex-col p-4">
          <div className="px-4 lg:px-6">
            {/* Data Setup Card */}
            <Card className="gap-4">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  <CardTitle>Data Setup</CardTitle>
                </div>
                <CardDescription>
                  Configure data source and load historical data
                </CardDescription>
              </CardHeader>

              <CardContent className="gap-4">
                <div className="gap-4 flex flex-col">
                  <Label>Data Source</Label>
                  <RadioGroup
                    value={dataSource}
                    onValueChange={setDataSource}
                    className="flex gap-4"
                  >
                    <RadioGroupItem value="realtime" id="realtime" />
                    <Label htmlFor="realtime" className="cursor-pointer">
                      Real Time Data
                    </Label>
                  </RadioGroup>
                </div>

                <div className="gap-4 flex flex-col pt-4">
                  <Label>Historical Days</Label>
                  <Input
                    id="days"
                    type="number"
                    min={90}
                    max={365}
                    value={days}
                    onChange={(e) => setDays(parseInt(e.target.value) || 90)}
                    disabled={isLoading || isTraining}
                    className="w-50"
                  />
                </div>

                <div className="flex items-center gap-4 pt-4">
                  <Button
                    onClick={handleLoadData}
                    disabled={isLoading || isTraining}
                    variant="outline"
                    className="w-50"
                  >
                    {isLoading ? (
                      <>
                        <Spinner className="h-4 w-4" />
                        Loading..
                      </>
                    ) : (
                      <>
                        <Database className="h-4 w-4" />
                        Load Data
                      </>
                    )}
                  </Button>

                  {dataLoaded && (
                    <Badge variant="default" className="gap-1">
                      <CheckCircle className="h-3 w-3" />
                      {dataRows} rows loaded
                    </Badge>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Training Card */}
            <Card className="gap-4 mt-6">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  <CardTitle>Training</CardTitle>
                </div>
                <CardDescription>
                  Train and optimize prediction models
                </CardDescription>
              </CardHeader>

              <CardContent className="">
                <div className="grid gap-4 md:grid-cols-2">
                  {/* Basic XGBoost */}
                  <div className="rounded-lg border p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">Basic XGBoost Model</h3>
                      {trainingResult?.basic_model && !isTraining && (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      )}
                    </div>

                    {/* Progress bar saat training */}
                    {isTraining && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span className="text-sm">
                            Training... {model1Progress}%
                          </span>
                        </div>
                        <Progress value={model1Progress} />
                      </div>
                    )}

                    {/* Results */}
                    {trainingResult?.basic_model && !isTraining ? (
                      <div className="space-y-1 flex flex-col gap-1">
                        <Badge variant="outline" className="w-fit">
                          RMSE: $
                          {trainingResult.basic_model.test_metrics?.rmse?.toFixed(
                            2
                          ) || "N/A"}
                        </Badge>
                        <Badge variant="outline" className="w-fit">
                          R²:{" "}
                          {trainingResult.basic_model.test_metrics?.r2?.toFixed(
                            4
                          ) || "N/A"}
                        </Badge>
                      </div>
                    ) : !isTraining ? (
                      <p className="text-sm text-muted-foreground">
                        Not trained yet
                      </p>
                    ) : null}
                  </div>

                  {/* Bayesian XGBoost */}
                  <div className="rounded-lg border p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">
                        Bayesian XGBoost Model
                      </h3>
                      {trainingResult?.bayesian_model && !isTraining && (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      )}
                    </div>

                    {/* Progress bar saat training */}
                    {isTraining && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span className="text-sm">
                            Training + Optimizing... {model2Progress}%
                          </span>
                        </div>
                        <Progress value={model2Progress} />
                      </div>
                    )}

                    {/* Results */}
                    {trainingResult?.bayesian_model && !isTraining ? (
                      <div className="space-y-1 flex flex-col gap-1">
                        <Badge variant="outline" className="w-fit">
                          RMSE: $
                          {trainingResult.bayesian_model.test_metrics?.rmse?.toFixed(
                            2
                          ) || "N/A"}
                        </Badge>
                        <Badge variant="outline" className="w-fit">
                          R²:{" "}
                          {trainingResult.bayesian_model.test_metrics?.r2?.toFixed(
                            4
                          ) || "N/A"}
                        </Badge>
                      </div>
                    ) : !isTraining ? (
                      <p className="text-sm text-muted-foreground">
                        Not trained yet
                      </p>
                    ) : null}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2 w-full pt-4">
                  <Button
                    onClick={handleStartTraining}
                    disabled={!dataLoaded || isTraining}
                    className="flex-1"
                  >
                    {isTraining ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Training...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Start Training
                      </>
                    )}
                  </Button>

                  {isTraining && (
                    <Button
                      onClick={handleStopTraining}
                      variant="destructive"
                      className="flex-1"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop
                    </Button>
                  )}

                  {trainingResult && !isTraining && (
                    <Button
                      variant="outline"
                      className="flex-1"
                      onClick={handleViewResults}
                    >
                      <BarChart3 className="mr-2 h-4 w-4" />
                      {showResults ? "Hide Results" : "View Results"}
                    </Button>
                  )}
                </div>

                {/* Detailed Results Section */}
                {showResults && trainingResult && (
                  <div className="mt-4 p-4 rounded-lg border bg-muted/50 space-y-3">
                    <h3 className="font-semibold">Training Results Details</h3>

                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Basic XGBoost</h4>
                        <div className="space-y-1 text-sm">
                          <p>
                            RMSE: $
                            {trainingResult.basic_model?.test_metrics?.rmse?.toFixed(
                              2
                            ) || "N/A"}
                          </p>
                          <p>
                            MAE: $
                            {trainingResult.basic_model?.test_metrics?.mae?.toFixed(
                              2
                            ) || "N/A"}
                          </p>
                          <p>
                            R² Score:{" "}
                            {trainingResult.basic_model?.test_metrics?.r2?.toFixed(
                              4
                            ) || "N/A"}
                          </p>
                          <p className="text-xs text-muted-foreground pt-1">
                            Training time:{" "}
                            {trainingResult.basic_model?.training_time?.toFixed(
                              2
                            )}
                            s
                          </p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">
                          Bayesian XGBoost
                        </h4>
                        <div className="space-y-1 text-sm">
                          <p>
                            RMSE: $
                            {trainingResult.bayesian_model?.test_metrics?.rmse?.toFixed(
                              2
                            ) || "N/A"}
                          </p>
                          <p>
                            MAE: $
                            {trainingResult.bayesian_model?.test_metrics?.mae?.toFixed(
                              2
                            ) || "N/A"}
                          </p>
                          <p>
                            R² Score:{" "}
                            {trainingResult.bayesian_model?.test_metrics?.r2?.toFixed(
                              4
                            ) || "N/A"}
                          </p>
                          <p className="text-xs text-muted-foreground pt-1">
                            Training time:{" "}
                            {trainingResult.bayesian_model?.training_time?.toFixed(
                              2
                            )}
                            s
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-2 border-t">
                      <h4 className="text-sm font-medium mb-2">
                        Best Bayesian Parameters
                      </h4>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <Badge variant="secondary">
                          Max Depth:{" "}
                          {
                            trainingResult.bayesian_model?.best_params
                              ?.max_depth
                          }
                        </Badge>
                        <Badge variant="secondary">
                          Learning Rate:{" "}
                          {trainingResult.bayesian_model?.best_params?.learning_rate?.toFixed(
                            4
                          )}
                        </Badge>
                        <Badge variant="secondary">
                          N Estimators:{" "}
                          {
                            trainingResult.bayesian_model?.best_params
                              ?.n_estimators
                          }
                        </Badge>
                        <Badge variant="secondary">
                          Subsample:{" "}
                          {
                            trainingResult.bayesian_model?.best_params
                              ?.subsample
                          }
                        </Badge>
                      </div>
                    </div>

                    <div className="pt-2 text-sm text-muted-foreground border-t">
                      <p>Training completed with {dataRows} rows of data</p>
                      <p>
                        Date range:{" "}
                        {trainingResult.data_info?.date_range?.start} to{" "}
                        {trainingResult.data_info?.date_range?.end}
                      </p>
                      <p>
                        Latest price: $
                        {trainingResult.data_info?.latest_price?.toFixed(2)}
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Training Log Card */}
            <Card className="mt-6">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  <CardTitle>Recent Training Log</CardTitle>
                </div>
                <CardDescription>
                  Real-time training process updates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {logs.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No logs yet
                    </p>
                  ) : (
                    logs.map((log, index) => (
                      <div
                        key={index}
                        className="rounded-lg bg-muted/50 px-4 py-2.5 text-sm hover:bg-muted transition-colors"
                      >
                        {log}
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
