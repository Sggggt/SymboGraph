import { AlertCircle, LoaderCircle } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function LoadingBlock({ rows = 3 }: { rows?: number }) {
  return (
    <div
      role="status"
      aria-live="polite"
      data-loading-rows={rows}
      className="flex min-h-44 w-full items-center justify-center gap-3 text-sm text-white/58"
    >
      <LoaderCircle className="size-6 animate-spin text-cyan-100" />
      <span>正在加载...</span>
    </div>
  );
}

export function ErrorBlock({ message }: { message: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertCircle className="size-4 text-destructive" />
          请求失败
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">{message}</p>
      </CardContent>
    </Card>
  );
}
