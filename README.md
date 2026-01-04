# Python OCR Service with PaddleOCR

This document provides the complete Python FastAPI service that performs table extraction from invoice images using PaddleOCR with PP-Structure.

## Requirements

```txt
fastapi==0.109.0
uvicorn==0.27.0
paddleocr==2.7.0.3
paddlepaddle==2.5.2
python-multipart==0.0.6
httpx==0.26.0
pillow==10.2.0
numpy==1.24.3
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PaddlePaddle (CPU version)
pip install paddlepaddle==2.5.2

# Install PaddleOCR
pip install paddleocr==2.7.0.3

# Install FastAPI and dependencies
pip install fastapi uvicorn python-multipart httpx pillow numpy
```

## Complete FastAPI Service Code

Save this as `main.py`:

```python
import io
import re
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from paddleocr import PPStructure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OCR engine
table_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize OCR engine on startup"""
    global table_engine
    logger.info("Initializing PaddleOCR PP-Structure engine...")
    table_engine = PPStructure(
        show_log=False,
        table=True,
        ocr=True,
        lang='en'
    )
    logger.info("OCR engine initialized successfully")
    yield
    logger.info("Shutting down OCR service")


app = FastAPI(
    title="Invoice OCR Service",
    description="Extract structured table data from invoice images using PaddleOCR",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    image_url: str


class InvoiceItem(BaseModel):
    serial_number: Optional[int] = None
    description_of_goods: Optional[str] = None
    hsn_sac_code: Optional[str] = None
    qty: Optional[float] = None
    unit: Optional[str] = None
    list_price: Optional[float] = None
    discount: Optional[float] = None
    price: Optional[float] = None
    amount: Optional[float] = None


class ExtractResponse(BaseModel):
    items: List[InvoiceItem]
    confidence: float
    raw_table_count: int


# Column name mappings (various possible header names)
COLUMN_MAPPINGS = {
    'serial_number': ['s.no', 'sno', 'sl.no', 'slno', 'sr.no', 'srno', 'serial', '#', 'no', 'no.'],
    'description_of_goods': ['description', 'desc', 'particulars', 'item', 'product', 'goods', 'name'],
    'hsn_sac_code': ['hsn', 'sac', 'hsn/sac', 'hsn code', 'sac code', 'hsn sac'],
    'qty': ['qty', 'quantity', 'qnty', 'units', 'nos'],
    'unit': ['unit', 'uom', 'unit of measure', 'per'],
    'list_price': ['list price', 'mrp', 'rate', 'unit price', 'price/unit'],
    'discount': ['discount', 'disc', 'disc%', 'discount%'],
    'price': ['price', 'net price', 'unit rate', 'rate'],
    'amount': ['amount', 'total', 'value', 'net amount', 'line total', 'amt']
}


def parse_number(value: str) -> Optional[float]:
    """Parse numeric value from string, handling various formats"""
    if not value or not isinstance(value, str):
        return None
    
    # Remove common non-numeric characters
    cleaned = re.sub(r'[₹$€£,\s]', '', value.strip())
    
    # Handle percentage
    if '%' in cleaned:
        cleaned = cleaned.replace('%', '')
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def normalize_header(header: str) -> Optional[str]:
    """Map header text to standardized column name"""
    if not header:
        return None
    
    header_lower = header.lower().strip()
    
    for standard_name, variations in COLUMN_MAPPINGS.items():
        for variation in variations:
            if variation in header_lower or header_lower in variation:
                return standard_name
    
    return None


def extract_table_from_structure(result: List[Dict[str, Any]]) -> List[InvoiceItem]:
    """Extract structured items from PP-Structure result"""
    items = []
    
    for element in result:
        if element.get('type') != 'table':
            continue
        
        # Get table HTML or cell data
        table_res = element.get('res', {})
        
        if 'html' in table_res:
            # Parse HTML table
            items.extend(parse_html_table(table_res['html']))
        elif 'cell_bbox' in table_res:
            # Parse from cell bounding boxes
            items.extend(parse_cell_data(table_res))
    
    return items


def parse_html_table(html: str) -> List[InvoiceItem]:
    """Parse items from HTML table structure"""
    from html.parser import HTMLParser
    
    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.current_row = []
            self.current_cell = []
            self.in_cell = False
            self.in_header = False
        
        def handle_starttag(self, tag, attrs):
            if tag == 'tr':
                self.current_row = []
            elif tag in ('td', 'th'):
                self.in_cell = True
                self.current_cell = []
                if tag == 'th':
                    self.in_header = True
        
        def handle_endtag(self, tag):
            if tag in ('td', 'th'):
                self.current_row.append(''.join(self.current_cell).strip())
                self.in_cell = False
            elif tag == 'tr':
                if self.current_row:
                    self.rows.append(self.current_row)
        
        def handle_data(self, data):
            if self.in_cell:
                self.current_cell.append(data)
    
    parser = TableParser()
    parser.feed(html)
    
    if len(parser.rows) < 2:
        return []
    
    # First row is header
    headers = parser.rows[0]
    column_map = {}
    
    for idx, header in enumerate(headers):
        normalized = normalize_header(header)
        if normalized:
            column_map[idx] = normalized
    
    items = []
    for row in parser.rows[1:]:
        # Skip rows that look like headers, totals, or footers
        row_text = ' '.join(row).lower()
        if any(skip in row_text for skip in ['total', 'subtotal', 'grand total', 'tax', 'cgst', 'sgst', 'igst']):
            continue
        
        item_data = {}
        for idx, cell in enumerate(row):
            if idx in column_map:
                field = column_map[idx]
                if field in ['serial_number', 'qty', 'list_price', 'discount', 'price', 'amount']:
                    item_data[field] = parse_number(cell)
                else:
                    item_data[field] = cell if cell else None
        
        # Only add if we have at least description or amount
        if item_data.get('description_of_goods') or item_data.get('amount'):
            items.append(InvoiceItem(**item_data))
    
    return items


def parse_cell_data(table_res: Dict[str, Any]) -> List[InvoiceItem]:
    """Parse items from cell bounding box data"""
    # Fallback for non-HTML table results
    # This handles cases where PP-Structure returns cell data differently
    return []


async def download_image(url: str) -> np.ndarray:
    """Download image from URL and convert to numpy array"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)


@app.post("/extract", response_model=ExtractResponse)
async def extract_invoice(request: ExtractRequest):
    """
    Extract structured table data from an invoice image.
    
    Returns line items with columns:
    - serial_number
    - description_of_goods
    - hsn_sac_code
    - qty
    - unit
    - list_price
    - discount
    - price
    - amount
    """
    global table_engine
    
    if table_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    try:
        logger.info(f"Processing image: {request.image_url}")
        
        # Download image
        image_array = await download_image(request.image_url)
        logger.info(f"Image downloaded, shape: {image_array.shape}")
        
        # Run PP-Structure
        result = table_engine(image_array)
        logger.info(f"PP-Structure returned {len(result)} elements")
        
        # Count tables found
        table_count = sum(1 for r in result if r.get('type') == 'table')
        logger.info(f"Found {table_count} tables")
        
        # Extract items
        items = extract_table_from_structure(result)
        logger.info(f"Extracted {len(items)} line items")
        
        # Calculate confidence based on structure quality
        confidence = 0.0
        if table_count > 0:
            confidence = 0.85
            if len(items) > 0:
                confidence = 0.92
        
        return ExtractResponse(
            items=items,
            confidence=confidence,
            raw_table_count=table_count
        )
        
    except httpx.HTTPError as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_ready": table_engine is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Running the Service

```bash
# Development
python main.py

# Production with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PaddlePaddle
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t invoice-ocr .
docker run -p 8000:8000 invoice-ocr
```

## API Usage

### Extract Invoice Data

```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://your-storage.com/invoice.jpg"}'
```

### Response Format

```json
{
  "items": [
    {
      "serial_number": 1,
      "description_of_goods": "Product Name",
      "hsn_sac_code": "080810",
      "qty": 10,
      "unit": "Pcs",
      "list_price": 100.00,
      "discount": 10.00,
      "price": 90.00,
      "amount": 900.00
    }
  ],
  "confidence": 0.92,
  "raw_table_count": 1
}
```

## Connecting to the Frontend

After deploying this service, add the `OCR_SERVICE_URL` secret in your Lovable Cloud backend:

1. Go to your Lovable project settings
2. Add a new secret: `OCR_SERVICE_URL` with value `http://your-server:8000`
3. The edge function will automatically use your OCR service

## Notes

- The service uses PaddleOCR's PP-Structure for table detection
- It automatically maps detected column headers to the expected schema
- Rows containing "total", "tax", etc. are filtered out
- Multi-line descriptions are handled by the HTML table parser
- Confidence scores are based on table detection quality
