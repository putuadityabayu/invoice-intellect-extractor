
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { InvoiceData } from "@/types/invoice";

interface InvoicePreviewProps {
  invoiceData: InvoiceData | null;
  isLoading: boolean;
}

export const InvoicePreview = ({ invoiceData, isLoading }: InvoicePreviewProps) => {
  const formatCurrency = (value: number | undefined) => {
    if (value === undefined) return "-";
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value);
  };

  // Function to handle empty state or loading state
  const renderEmptyState = () => (
    <div className="flex flex-col items-center justify-center h-64 text-center p-4">
      <p className="text-gray-500 mb-2">No invoice data yet</p>
      <p className="text-sm text-gray-400">
        Upload an invoice image to see the extracted data here
      </p>
    </div>
  );

  // Function to render loading skeletons
  const renderLoadingState = () => (
    <div className="space-y-4 p-4">
      <Skeleton className="h-6 w-3/4" />
      <Skeleton className="h-6 w-1/2" />
      <Separator className="my-4" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <Separator className="my-4" />
      <Skeleton className="h-4 w-1/2" />
      <Skeleton className="h-4 w-1/3" />
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Extracted Data</CardTitle>
        <CardDescription>
          Structured data extracted from the invoice
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          renderLoadingState()
        ) : !invoiceData ? (
          renderEmptyState()
        ) : (
          <div className="space-y-6">
            {/* Header Information */}
            <div className="space-y-2">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-medium text-gray-500">Invoice Number</p>
                  <p className="font-semibold">{invoiceData.invoice_number || "N/A"}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-500">Date</p>
                  <p className="font-medium">{invoiceData.invoice_date || "N/A"}</p>
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Customer</p>
                <p className="font-medium">{invoiceData.name || "N/A"}</p>
              </div>
            </div>

            <Separator />

            {/* Items */}
            <div>
              <p className="text-sm font-medium text-gray-500 mb-2">Items</p>
              {invoiceData.items && invoiceData.items.length > 0 ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-12 text-xs font-medium text-gray-500">
                    <div className="col-span-5">Description</div>
                    <div className="col-span-2 text-right">Qty</div>
                    <div className="col-span-2 text-right">Price</div>
                    <div className="col-span-3 text-right">Total</div>
                  </div>
                  {invoiceData.items.map((item, index) => (
                    <div key={index} className="grid grid-cols-12 text-sm">
                      <div className="col-span-5 font-medium truncate">
                        {item.name || "Unnamed item"}
                      </div>
                      <div className="col-span-2 text-right">
                        {item.quantity || "-"}
                      </div>
                      <div className="col-span-2 text-right">
                        {formatCurrency(item.unit_price)}
                      </div>
                      <div className="col-span-3 text-right font-medium">
                        {formatCurrency(item.total_price)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-400 italic">No items found</p>
              )}
            </div>

            <Separator />

            {/* Totals */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <p className="text-sm">Subtotal</p>
                <p className="text-sm font-medium">{formatCurrency(invoiceData.subtotal)}</p>
              </div>

              {/* Extra prices (tax, shipping, etc.) */}
              {invoiceData.extra_price &&
                invoiceData.extra_price.map((extra, index) => {
                  const [key, value] = Object.entries(extra)[0];
                  return (
                    <div key={index} className="flex justify-between">
                      <p className="text-sm capitalize">{key}</p>
                      <p className="text-sm font-medium">{formatCurrency(value)}</p>
                    </div>
                  );
                })}

              <div className="flex justify-between font-bold">
                <p>Total</p>
                <p>{formatCurrency(invoiceData.total)}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
