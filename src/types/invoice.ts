
export interface InvoiceItem {
  name: string;
  quantity: number;
  unit_price: number;
  total_price: number;
}

export interface ExtraPrice {
  [key: string]: number;
}

export interface InvoiceData {
  invoice_number: string;
  invoice_date: string;
  name: string;
  items: InvoiceItem[];
  subtotal: number;
  extra_price: ExtraPrice[];
  total: number;
}
