
import { useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { FileUpload } from "@/components/FileUpload";
import { InvoicePreview } from "@/components/InvoicePreview";
import { InvoiceData } from "@/types/invoice";
import { UploadCloud, Link2 } from "lucide-react";

const Index = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const [invoiceData, setInvoiceData] = useState<InvoiceData | null>(null);
  const [selectedTab, setSelectedTab] = useState("file");
  
  // Backend URL - would be configured in environment variables in a real app
  const backendUrl = "http://localhost:5000";

  const handleImageUrlSubmit = async () => {
    if (!imageUrl.trim()) {
      toast({
        title: "Error",
        description: "Please enter a valid image URL",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("image_url", imageUrl);
      
      const response = await fetch(`${backendUrl}/extract`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setInvoiceData(data);
      toast({
        title: "Success",
        description: "Invoice data extracted successfully",
      });
    } catch (error) {
      console.error("Error extracting invoice data:", error);
      toast({
        title: "Error",
        description: "Failed to extract invoice data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const response = await fetch(`${backendUrl}/extract`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setInvoiceData(data);
      toast({
        title: "Success",
        description: "Invoice data extracted successfully",
      });
    } catch (error) {
      console.error("Error extracting invoice data:", error);
      toast({
        title: "Error",
        description: "Failed to extract invoice data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-8 max-w-5xl">
      <h1 className="text-3xl font-bold mb-2 text-center">Invoice Intellect Extractor</h1>
      <p className="text-gray-500 mb-8 text-center">Upload an invoice image to extract structured data</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Upload Invoice</CardTitle>
              <CardDescription>
                Upload an invoice image or provide a URL to extract data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={selectedTab} onValueChange={setSelectedTab}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="file">File Upload</TabsTrigger>
                  <TabsTrigger value="url">Image URL</TabsTrigger>
                </TabsList>
                <TabsContent value="file" className="mt-4">
                  <FileUpload onUpload={handleFileUpload} isLoading={loading} />
                </TabsContent>
                <TabsContent value="url" className="mt-4">
                  <div className="space-y-4">
                    <div className="grid w-full gap-1.5">
                      <Label htmlFor="imageUrl">Image URL</Label>
                      <div className="flex w-full items-center space-x-2">
                        <Input
                          id="imageUrl"
                          placeholder="https://example.com/invoice.jpg"
                          value={imageUrl}
                          onChange={(e) => setImageUrl(e.target.value)}
                          disabled={loading}
                        />
                        <Button 
                          type="submit" 
                          onClick={handleImageUrlSubmit}
                          disabled={loading || !imageUrl.trim()}
                        >
                          <Link2 className="h-4 w-4 mr-2" />
                          Extract
                        </Button>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
            <CardFooter className="flex justify-between">
              <p className="text-sm text-gray-500">
                Supported formats: JPG, PNG, PDF
              </p>
            </CardFooter>
          </Card>
        </div>
        
        <div>
          <InvoicePreview invoiceData={invoiceData} isLoading={loading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
