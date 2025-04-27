# main.py
from fastapi import FastAPI
from customers import segment_customers
from products import segment_products
from suppliers import segment_suppliers
from countries import segment_countries

app = FastAPI()

@app.get("/")
def home():
    return {"message": "DBSCAN Segmentasyon API - Hoşgeldiniz"}

@app.get("/segment/customers")
def customer_segmentation():
    return segment_customers()

@app.get("/segment/products")
def product_segmentation():
    return segment_products()

@app.get("/segment/suppliers")
def supplier_segmentation():
    return segment_suppliers()

@app.get("/segment/countries")
def country_segmentation():
    return segment_countries()
