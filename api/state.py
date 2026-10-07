"""Trạng thái dùng chung giữa main.py và các router (tránh import vòng)."""
services: dict = {}  # "main": RetrievalService, "space": SpaceProjector
