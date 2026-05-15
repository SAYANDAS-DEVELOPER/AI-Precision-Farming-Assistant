import os

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from deep_translator import GoogleTranslator
import requests
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import pymongo
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
import bcrypt
import datetime
import base64
import io
import threading
import wave
# Imports for timezone and MongoDB ObjectId
from pytz import timezone
from bson.objectid import ObjectId
# Imports for the robust Speech-to-Text feature
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import streamlit.components.v1 as components
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


st.set_page_config(page_title="KrishiSakhi", page_icon=Image.open("ICON.jpg"), layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* ===================================================
       KRISHISAKHI — UPGRADED UI  (drop-in CSS replacement)
       All original functionality preserved; only visuals
       are enhanced.
    =================================================== */

    /* ── Google Font import (Segoe UI fallback kept) ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&display=swap');

    /* ===== App-wide base styles ===== */
    html, body, [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 45%, #fffde7 100%);
        font-family: "DM Sans", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding: 28px 32px;
        border-radius: 18px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.08);
    }

    /* ===================================================
       SIDEBAR — Full Upgrade
    =================================================== */

    /* Sidebar container */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"],
    aside[data-testid="stSidebar"],
    nav[aria-label="Sidebar"] {
        background: linear-gradient(160deg, #ffffff 0%, #f1f8e9 60%, #e8f5e9 100%) !important;
        border-radius: 0 22px 22px 0 !important;
        padding: 0 !important;
        box-shadow: 4px 0 28px rgba(46,125,50,0.13), 0 12px 28px rgba(0,0,0,0.07) !important;
        display: block !important;
        visibility: visible !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 300px !important;
        position: relative !important;
        left: 0 !important;
        transform: none !important;
        opacity: 1 !important;
        border-right: 1.5px solid rgba(46,125,50,0.10) !important;
        overflow: hidden !important;
    }

    /* Inner scrollable content area */
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebar"] > div:first-child {
        padding: 0 16px 24px 16px !important;
        height: 100% !important;
        overflow-y: auto !important;
    }

    /* Thin green top accent bar */
    section[data-testid="stSidebar"]::before,
    div[data-testid="stSidebar"]::before {
        content: '';
        display: block;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #43a047, #1e88e5, #43a047);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
        flex-shrink: 0;
    }

    @keyframes shimmer {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Logo / brand area injected via ::after on a wrapper — 
       we style the first markdown block as a logo area */
    section[data-testid="stSidebar"] .stMarkdown:first-of-type,
    div[data-testid="stSidebar"] .stMarkdown:first-of-type {
        background: linear-gradient(135deg, rgba(46,125,50,0.08), rgba(30,136,229,0.06));
        border-radius: 14px;
        padding: 14px 16px 12px !important;
        margin: 12px 0 8px !important;
        border: 1px solid rgba(46,125,50,0.12);
    }

    /* All sidebar text */
    section[data-testid="stSidebar"] *,
    div[data-testid="stSidebar"] *,
    aside[data-testid="stSidebar"] *,
    nav[aria-label="Sidebar"] * {
        color: #2e7d32 !important;
        font-weight: 600;
        font-family: "DM Sans", "Segoe UI", Tahoma, sans-serif !important;
    }

    /* Sidebar radio buttons — nav items */
    section[data-testid="stSidebar"] .stRadio > div,
    div[data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }

    section[data-testid="stSidebar"] .stRadio label,
    div[data-testid="stSidebar"] .stRadio label {
        display: flex !important;
        align-items: center !important;
        padding: 9px 14px !important;
        border-radius: 12px !important;
        transition: background 0.2s ease, box-shadow 0.2s ease !important;
        cursor: pointer !important;
        border: 1px solid transparent !important;
        margin-bottom: 2px !important;
        font-size: 0.88rem !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover,
    div[data-testid="stSidebar"] .stRadio label:hover {
        background: linear-gradient(135deg, rgba(67,160,71,0.12), rgba(30,136,229,0.10)) !important;
        border-color: rgba(46,125,50,0.18) !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.10) !important;
    }

    /* Active/checked radio = active nav item */
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked),
    div[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, rgba(67,160,71,0.18), rgba(30,136,229,0.15)) !important;
        border-color: rgba(46,125,50,0.30) !important;
        box-shadow: 0 4px 16px rgba(46,125,50,0.15), inset 3px 0 0 #43a047 !important;
    }

    /* Sidebar selectbox (language picker) */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    div[data-testid="stSidebar"] .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid rgba(46,125,50,0.25) !important;
        background: rgba(255,255,255,0.85) !important;
        box-shadow: 0 2px 8px rgba(46,125,50,0.08) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div:hover,
    div[data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: rgba(46,125,50,0.50) !important;
        box-shadow: 0 4px 14px rgba(46,125,50,0.14) !important;
    }

    /* Sidebar section headers (st.sidebar.markdown("### ...")) */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: rgba(46,125,50,0.55) !important;
        font-weight: 700 !important;
        margin: 18px 0 6px !important;
        padding-left: 4px !important;
    }

    /* Sidebar logout / action buttons */
    section[data-testid="stSidebar"] .stButton > button,
    div[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%) !important;
        color: #2e7d32 !important;
        border: 1.5px solid rgba(46,125,50,0.35) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        width: 100% !important;
        margin-top: 4px !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.10) !important;
        transition: transform 0.15s ease, box-shadow 0.2s ease, background 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover,
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #43a047 0%, #1e88e5 100%) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 24px rgba(46,125,50,0.22) !important;
    }

    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr,
    div[data-testid="stSidebar"] hr {
        border: none !important;
        border-top: 1px solid rgba(46,125,50,0.15) !important;
        margin: 14px 0 !important;
    }

    /* Sidebar user info / caption text */
    section[data-testid="stSidebar"] .stCaption,
    div[data-testid="stSidebar"] .stCaption {
        color: rgba(46,125,50,0.60) !important;
        font-size: 0.78rem !important;
        font-weight: 400 !important;
    }

    /* Sidebar profile picture */
    section[data-testid="stSidebar"] img,
    div[data-testid="stSidebar"] img {
        border-radius: 50% !important;
        border: 2.5px solid rgba(46,125,50,0.30) !important;
        box-shadow: 0 4px 14px rgba(46,125,50,0.18) !important;
    }

    /* Glow pulse on sidebar avatar on hover */
    section[data-testid="stSidebar"] img:hover,
    div[data-testid="stSidebar"] img:hover {
        box-shadow: 0 0 0 4px rgba(67,160,71,0.20), 0 6px 20px rgba(46,125,50,0.22) !important;
        transition: box-shadow 0.3s ease !important;
    }

    /* Sidebar text input (e.g. search) */
    section[data-testid="stSidebar"] input,
    div[data-testid="stSidebar"] input {
        border-radius: 10px !important;
        border: 1.5px solid rgba(46,125,50,0.25) !important;
        background: rgba(255,255,255,0.9) !important;
    }
    section[data-testid="stSidebar"] input:focus,
    div[data-testid="stSidebar"] input:focus {
        border-color: #43a047 !important;
        box-shadow: 0 0 0 3px rgba(67,160,71,0.15) !important;
    }

    /* Scrollbar inside sidebar */
    section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar,
    div[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        width: 4px;
    }
    section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track,
    div[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
        background: transparent;
    }
    section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb,
    div[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
        background: rgba(46,125,50,0.25);
        border-radius: 4px;
    }

    div[data-testid="stAppViewContainer"] > .main {
        margin-left: 300px !important;
    }

    /* ===================================================
       MAIN CONTENT
    =================================================== */

    /* ===== Header / Hero ===== */
    .hero {
        background: linear-gradient(135deg, rgba(46,125,50,0.92), rgba(3,169,244,0.88));
        border-radius: 20px;
        padding: 32px 28px;
        color: #ffffff;
        box-shadow: 0 16px 48px rgba(46,125,50,0.22), 0 4px 16px rgba(0,0,0,0.10);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        animation: heroFadeIn 0.7s ease both;
    }

    @keyframes heroFadeIn {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Decorative orb */
    .hero::after {
        content: '';
        position: absolute;
        top: -50px; right: -50px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: rgba(255,255,255,0.07);
        animation: orbFloat 5s ease-in-out infinite;
        pointer-events: none;
    }
    .hero::before {
        content: '';
        position: absolute;
        bottom: -60px; right: 90px;
        width: 150px; height: 150px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
        animation: orbFloat 5s 1.8s ease-in-out infinite;
        pointer-events: none;
    }
    @keyframes orbFloat {
        0%, 100% { transform: scale(1) translateY(0);   }
        50%       { transform: scale(1.15) translateY(-8px); }
    }

    .hero h1 {
        font-size: 2.6rem;
        margin: 0 0 8px;
        letter-spacing: 0.02em;
        font-weight: 700;
        position: relative; z-index: 1;
    }
    .hero p {
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 820px;
        opacity: 0.93;
        position: relative; z-index: 1;
    }

    /* ===== Feature Cards ===== */
    .feature-card {
        background: rgba(255,255,255,0.95);
        border-radius: 18px;
        padding: 20px 20px 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.07);
        border: 1px solid rgba(46,125,50,0.10);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        position: relative;
        overflow: hidden;
    }
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #43a047, #1e88e5);
        opacity: 0;
        transition: opacity 0.22s ease;
        border-radius: 18px 18px 0 0;
    }
    .feature-card:hover {
        transform: translateY(-7px);
        box-shadow: 0 20px 48px rgba(46,125,50,0.14), 0 4px 16px rgba(0,0,0,0.08);
        border-color: rgba(46,125,50,0.22);
    }
    .feature-card:hover::before {
        opacity: 1;
    }
    .feature-card h4 {
        margin: 0 0 10px;
        color: #1b5e20;
        font-size: 1rem;
        font-weight: 700;
    }
    .feature-card p {
        margin: 0 0 14px;
        color: #37474f;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* ===== Titles & Text ===== */
    h1, h2, h3 {
        color: #1b5e20 !important;
        font-weight: 700;
    }

    p, li, span, div[data-testid="stMarkdownContainer"] {
        color: #212121;
        font-size: 15px;
    }

    /* ===== Metrics Styling ===== */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92) !important;
        border-radius: 16px !important;
        padding: 18px 16px !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.07) !important;
        border: 1px solid rgba(46,125,50,0.10) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 14px 36px rgba(46,125,50,0.13) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1565c0 !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #f57f17 !important;
        font-weight: 600 !important;
        font-size: 0.80rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    /* ===== Tables/Dataframes ===== */
    .stDataFrame {
        background: #ffffff;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid rgba(46,125,50,0.08);
    }

    /* ===== Buttons (main content) ===== */
    .stButton > button {
        background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%) !important;
        color: #0d47a1 !important;
        border: 2px solid rgba(13,71,161,0.85) !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-family: "DM Sans", "Segoe UI", sans-serif !important;
        box-shadow: 0 6px 16px rgba(13,71,161,0.15) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease, filter 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 30px rgba(13,71,161,0.25) !important;
        filter: drop-shadow(0 0 10px rgba(30,136,229,0.40)) !important;
        border-color: transparent !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ===== Top Navigation Bar ===== */
    .nav-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 18px;
        justify-content: center;
    }
    .nav-button {
        background: linear-gradient(135deg, #43a047, #1e88e5);
        border: none;
        border-radius: 999px;
        color: white;
        cursor: pointer;
        font-weight: 700;
        font-family: "DM Sans", "Segoe UI", sans-serif;
        padding: 10px 18px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        box-shadow: 0 8px 22px rgba(67,160,71,0.25);
        font-size: 0.84rem;
    }
    .nav-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px rgba(67,160,71,0.32);
        filter: brightness(1.05);
    }
    .nav-button:active {
        transform: translateY(0);
    }

    /* ===== Form inputs ===== */
    input[type="text"], input[type="password"], textarea,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 1.5px solid rgba(46,125,50,0.22) !important;
        background: rgba(255,255,255,0.92) !important;
        font-family: "DM Sans", "Segoe UI", sans-serif !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    input[type="text"]:focus, input[type="password"]:focus, textarea:focus,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #43a047 !important;
        box-shadow: 0 0 0 3px rgba(67,160,71,0.15) !important;
        outline: none !important;
    }

    /* ===== Selectbox ===== */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid rgba(46,125,50,0.22) !important;
        background: rgba(255,255,255,0.92) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(46,125,50,0.45) !important;
        box-shadow: 0 0 0 3px rgba(67,160,71,0.10) !important;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.70) !important;
        border-radius: 14px !important;
        padding: 4px !important;
        gap: 4px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
        border: 1px solid rgba(46,125,50,0.10) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: "DM Sans", "Segoe UI", sans-serif !important;
        transition: background 0.2s, color 0.2s !important;
        color: rgba(46,125,50,0.65) !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(67,160,71,0.16), rgba(30,136,229,0.14)) !important;
        color: #1b5e20 !important;
        box-shadow: 0 2px 10px rgba(46,125,50,0.12) !important;
    }

    /* ===== Expander ===== */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.85) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(46,125,50,0.15) !important;
        font-weight: 700 !important;
        color: #1b5e20 !important;
        transition: background 0.2s, box-shadow 0.2s !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(255,255,255,1) !important;
        box-shadow: 0 4px 16px rgba(46,125,50,0.12) !important;
    }
    .streamlit-expanderContent {
        border: 1px solid rgba(46,125,50,0.10) !important;
        border-top: none !important;
        border-radius: 0 0 14px 14px !important;
        background: rgba(255,255,255,0.70) !important;
    }

    /* ===== Alerts / Success / Warning / Error ===== */
    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border-left-width: 4px !important;
        font-weight: 500 !important;
    }

    /* ===== Spinner ===== */
    .stSpinner > div {
        border-top-color: #43a047 !important;
    }

    /* ===== File uploader ===== */
    .stFileUploader > div {
        border-radius: 16px !important;
        border: 2px dashed rgba(46,125,50,0.28) !important;
        background: rgba(255,255,255,0.80) !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    .stFileUploader > div:hover {
        border-color: rgba(67,160,71,0.55) !important;
        background: rgba(232,245,233,0.60) !important;
    }

    /* ===== Floating home button ===== */
    .floating-home {
        position: fixed;
        right: 82px;
        bottom: 20px;
        z-index: 9999;
        background: linear-gradient(135deg, #43a047, #1e88e5);
        color: white;
        padding: 12px 18px;
        border-radius: 999px;
        border: none;
        font-size: 14px;
        font-weight: 700;
        font-family: "DM Sans", "Segoe UI", sans-serif;
        box-shadow: 0 12px 28px rgba(67,160,71,0.30);
        cursor: pointer;
        transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
    }
    .floating-home:hover {
        transform: translateY(-2px);
        opacity: 0.95;
        box-shadow: 0 18px 38px rgba(67,160,71,0.38);
    }

    /* ===== Floating Chatbot Launcher ===== */
    .chat-widget {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        font-family: "DM Sans", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-toggle-checkbox {
        display: none;
    }
    .chat-launcher {
        width: 52px;
        height: 52px;
        background: linear-gradient(135deg, #43a047 0%, #1e88e5 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 12px 32px rgba(46,125,50,0.30);
        cursor: pointer;
        transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease;
        border: none;
    }
    .chat-launcher:hover {
        transform: scale(1.12) translateY(-3px);
        box-shadow: 0 20px 44px rgba(46,125,50,0.38);
    }
    .chat-panel {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: min(420px, 92vw);
        max-height: 82vh;
        background: #ffffff;
        border-radius: 22px;
        box-shadow: 0 22px 62px rgba(0,0,0,0.18);
        opacity: 0;
        visibility: hidden;
        transform: translateY(14px);
        transition: opacity 0.28s ease, transform 0.28s ease, visibility 0.28s ease;
        overflow: hidden;
    }
    .chat-toggle-checkbox:checked + .chat-launcher + .chat-panel {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
    .chat-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 18px;
        background: linear-gradient(135deg, #43a047 0%, #1e88e5 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.01em;
    }
    .chat-panel-header button,
    .chat-panel-header label {
        background: transparent;
        border: none;
        color: #ffffff;
        font-size: 20px;
        cursor: pointer;
        outline: none;
    }
    .chat-panel-body {
        padding: 0;
    }
    .chat-panel iframe {
        width: 100%;
        height: 520px;
        border: none;
        display: block;
    }
    @media (max-width: 600px) {
        .chat-panel {
            width: calc(100vw - 24px);
            right: 12px;
            bottom: 90px;
        }
        .chat-panel iframe {
            height: 420px;
        }
    }

    /* ===== Hide Streamlit default menu/footer ===== */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================================
# 1. TRANSLATION AND LOCALIZATION SETUP (Updated with Community Forum keys)
# =====================================================================================

translations = {
    'en': {
        'app_title': "KrishiSakhi", 'go_to': "Go to", 'go_button': "Go", 'back_to_home': "Back to Home", 'language_selector_label': "Language / भाषा",
        'page_home': "Home", 'page_crop_recommendation': "Crop Recommendation", 'page_yield_prediction': "Yield Prediction", 'page_crop_calendar': "Crop Calendar", 'page_weather_report': "Weather Report", 'page_leaf_disease_prediction': "Leaf Disease Prediction", 'page_translator': "Translator", 'page_profile': "Profile", 'page_logs': "Logs", 'page_community_forum': "Community Forum",
        'quick_actions_title': "🚀 Quick Actions", 'quick_action_crop_desc': "Get tailored crop suggestions based on your soil and weather.", 'quick_action_weather_desc': "Check the weather in your area and plan your farming activities.",
        'page_crop_guide': "Crop Guide", 'page_pest_prediction': "Pest Prediction", 'page_soil_type_prediction': "Soil Type Prediction",
        'crop_guide_title': "Crop Fertigation and Irrigation Guide", 'select_stage_label': "Select a Growth Stage", 'guide_table_header': "Guidance Details",
        'file_not_found_error': "The required fertigation/irrigation file was not found. Please ensure it is in the correct folder.",
        'home_title': "Welcome to KrishiSakhi! 🌱", 'home_subtitle': "Your AI-powered assistant for modern farming.",
        'home_intro': "*KrishiSakhi* is designed to empower farmers with data-driven insights.\nNavigate through the sections in the sidebar to access different features:\n- *🌾 Crop Recommendation:* Get suggestions for the best crops to plant based on soil and environmental factors.\n- *📈 Yield Prediction:* Forecast your crop yield based on historical data and farming parameters.\n- *🗓 Crop Calendar:* View the ideal sowing and harvesting times for various crops.\n- *🌦 Weather Report:* Get real-time weather updates and forecasts for your location.\n- *🌿 Leaf Disease Prediction:* Upload a leaf image to detect diseases early.\n- *🌐 Translator:* Translate farming terms and sentences between different Indian languages.\n- *👥 Community Forum:* Connect with other farmers, ask questions, and share solutions.",
        'home_tab_crop_management': "🌾 Crop Management", 'home_tab_analytics': "📊 Analytics & Prediction", 'home_tab_environment': "🌦 Environment", 'home_tab_community': "🤝 Community & Tools", 'home_section_crop_planning': "### Crop Planning & Management", 'home_section_data_insights': "### Data-Driven Insights", 'home_section_environment': "### Environmental Monitoring", 'home_section_community': "### Community & Utilities", 'home_card_crop_calendar_desc': "View optimal sowing and harvesting times for crops.", 'home_card_yield_desc': "Forecast crop yield based on historical data and parameters.", 'home_card_soil_desc': "Identify soil type and properties from images.", 'home_card_pest_desc': "Identify pests from images for timely intervention.", 'home_card_translator_desc': "Translate farming terms across Indian languages.", 'home_about_header': "📖 About KrishiSakhi", 'ai_chatbot_assistant_header': "### 🤖 AI Chatbot Assistant", 'ai_chatbot_assistant_desc': "Get instant answers to your farming questions:",
        'crop_rec_header': "🌾 Crop Recommendation", 'crop_rec_intro': "Enter the following details to get a crop recommendation.",
        'nitrogen_label': "Nitrogen (N)", 'phosphorus_label': "Phosphorus (P)", 'potassium_label': "Potassium (K)", 'temperature_label': "Temperature (°C)", 'humidity_label': "Humidity (%)", 'ph_label': "pH Value", 'rainfall_label': "Rainfall (mm)",
        'recommend_crop_button': "Recommend Crop", 'recommendation_success': "🎉 The recommended crop is *{crop}*.",
        'crop_info_header': "More Information about {crop}", 'description_heading': "📖 Description", 'irrigation_heading': "💧 Irrigation Tips", 'conditions_heading': "☀️ Ideal Conditions", 'pests_heading': "🐛 Common Pests & Diseases",
        'yield_pred_header': "📈 Crop Yield Prediction", 'yield_pred_intro': "Fill in the details below to predict the crop yield for a specific area.",
        'year_label': "Year", 'crop_label': "Crop", 'season_label': "Season", 'area_label': "Area (in Hectares)",
        'predict_yield_button': "Predict Yield", 'yield_prediction_success': "🌾 The predicted crop yield is approximately *{yield_val:.2f} kg/ha*.",
        'market_price_header': "💰 Estimated Market Value", 'fetching_prices_spinner': "Fetching latest market prices...", 'market_price_label': "Avg. Market Price (per Quintal)", 'total_value_label': "Total Estimated Value", 'profit_header': "💵 Estimated Profit", 'cost_per_ha_label': "Farming Cost per Hectare (₹)", 'total_cost_label': "Total Farming Cost", 'total_profit_label': "Total Estimated Profit",
        'price_not_available': "Price data for {crop} is currently unavailable.", 'price_api_error': "Could not fetch market price data at this time.",
        'crop_calendar_header': "🗓 Crop Sowing and Harvest", 'select_crop_label': "Select a Crop",
        'schedule_for_crop': "📅 Schedule for {crop}", 'sowing_start_label': "Sowing Start", 'sowing_end_label': "Sowing End", 'harvest_start_label': "Harvest Start", 'harvest_end_label': "Harvest End", 'fertiliser_label': "Fertiliser (N:P:K kg/ha)", 'duration_label': "Duration",
        'weather_report_header': "🌦 Weather Report", 'enter_city_label': "Enter the name of your city", 'get_weather_button': "Get Weather",
        'current_weather_in': "Current Weather in {location}", 'temperature_metric': "Temperature", 'feels_like_metric': "Feels Like", 'humidity_metric': "Humidity", 'description_label': "Description:", 'wind_label': "Wind:", 'precipitation_label': "Precipitation:",
        'forecast_header': "🗓 3-Day Forecast", 'avg_temp_label': "Avg Temp:", 'max_temp_label': "Max Temp:", 'min_temp_label': "Min Temp:", 'sunrise_label': "Sunrise:", 'sunset_label': "Sunset:", 'outlook_label': "Outlook:",
        'error_parsing_weather': "Could not parse weather data. Please check the city name and try again. Error: {e}", 'enter_city_warning': "⚠ Please enter a city name.",
        'weather_fetch_error': "Could not retrieve weather data. Please check your network connection or the city name and try again.", 'weather_api_key_error': "API Key Error: {message}. Please check your OpenWeatherMap API key in secrets.toml. Note that new keys can take a few hours to activate.", 'weather_city_not_found_error': "City Not Found Error: {message}. Please check the spelling of the city name.", 'weather_api_generic_error': "API Error ({cod}): {message}",
        'translator_header': "🌐 Farmer Language Translator", 'translator_intro': "✍ Enter text to translate", 'translate_button': "🔄 Translate", 'translation_failed_error': "❌ Translation failed: {e}", 'enter_text_warning': "⚠ Please enter some text.",
        'ai_chatbot_header': "🤖 AI Chatbot", 'ai_chatbot_intro': "This feature is coming soon! Stay tuned for a powerful AI chatbot to help with your farming questions.",
        'languages_supported_label': "Languages Supported", 'farmers_helped_label': "Farmers Helped", 'crops_covered_label': "Crops Covered", 'recommendations_given_label': "AI Recommendations",
        'leaf_disease_header': "🌿 Leaf Disease Prediction", 'leaf_disease_intro': "Upload an image of a plant leaf to check for diseases.", 'model_not_found_warning': "⚠ Disease prediction model not found. Please contact the administrator.",
        'uploader_label': "Choose a leaf image...", 'uploaded_image_caption': "Uploaded Image.", 'predict_disease_button': "🔍 Predict Disease",
        'predicting_spinner': "Analyzing the leaf...", 'prediction_result': "✅ Prediction: *{disease}*",
        'remedy_info_header': "🩺 Treatment and Management for {disease}", 'disease_description_heading': "ℹ️ About the Disease", 'organic_remedies_heading': "🌿 Organic Remedies", 'chemical_remedies_heading': "🧪 Chemical Remedies", 'healthy_plant_message': "Great news! Your plant appears to be healthy. Continue with good agricultural practices.",
        'soil_prediction_header': "🏞️ Soil Type Prediction", 'soil_prediction_intro': "Upload an image of the soil to identify its type and learn about its properties.", 'upload_soil_image_label': "Choose a soil image...",
        'predict_soil_button': "🔍 Predict Soil Type", 'analyzing_soil_spinner': "Analyzing the soil...", 'soil_prediction_result': "✅ The predicted soil type is: *{soil_type}*", 'soil_info_header': "Information about {soil_type}",
        'pest_prediction_header': "Pest Identification from Image", 'upload_image_pest': "Upload an image of the pest or affected leaf", 'prediction_pest': "Predicted Pest / Insect", 'model_loading_error': "Could not load the pest prediction model. Please check the internet connection.",
        'remedy_info_header_pest': "🩺 Treatment and Management for {pest}", 'pest_description_heading': "ℹ️ About the Pest", 'no_remedy_info': "Remedy information for {pest} is not yet available.",
        'feedback_header': "Feedback", 'rating_label': "Rate your experience:", 'submit_button': "Submit", 'feedback_success': "Thanks for your feedback!",
        'contact_header': "Contact Us",
        'login_page_title': "Welcome to KrishiSakhi 🌱", 'login_tab': "Login", 'signup_tab': "Sign Up",
        'username_label': "Username", 'password_label': "Password", 'login_button': "Login", 'signup_button': "Sign Up", 'logout_button': "Logout",
        'login_error': "Incorrect username or password.", 'signup_success': "Registration successful! Please login.", 'signup_error_exists': "Username already exists. Please choose another one.", 'signup_error_general': "An error occurred during registration.",
        'welcome_back': "Welcome back, {username}!",
        'profile_header': "👤 My Profile", 'profile_subheader': "Manage your account details and settings.",
        'full_name_label': "Full Name", 'contact_number_label': "Contact Number", 'new_username_label': "New Username", 'new_password_label': "New Password", 'confirm_password_label': "Confirm New Password", 'profile_pic_label': "Profile Picture",
        'update_profile_button': "Update Profile", 'profile_update_success': "✅ Profile updated successfully!", 'profile_update_error': "❌ Could not update profile. Username may already exist.", 'password_mismatch_error': "Passwords do not match.",
        'danger_zone_header': "🚨 Danger Zone", 'delete_account_button': "Delete My Account", 'delete_account_warning': "This action is irreversible. All your data will be permanently deleted.", 'delete_confirmation_label': "To confirm, please type `DELETE` in the box below.",
        'delete_account_confirm_button': "Yes, I want to delete my account", 'account_deleted_success': "Your account has been successfully deleted.", 'incorrect_delete_confirmation': "Incorrect confirmation text.",
        'logs_header': "My Farming Logs", 'logs_intro': "Record your daily activities, observations, and notes here.", 'new_log_label': "New Log Entry", 'save_log_button': "Save Log", 'log_saved_success': "Log entry saved successfully!",
        'past_logs_header': "📖 Past Entries", 'no_logs_message': "You have no log entries yet.", 'delete_log_button': "Delete", 'log_deleted_success': "Log deleted!",
        'record_log_prompt': "Attach a voice note:", 'voice_note_recorded': "🎤 Voice note recorded and ready to be saved with your log.", 'attached_voice_note': "Attached Voice Note:", 'upload_media_label': "Attach photos or videos:",
        'attached_images': "Attached Images:", 'attached_videos': "Attached Videos:",
        'forum_header': "👥 Community Forum", 'forum_intro': "Connect with other farmers. Ask questions, share solutions, and grow together.",
        'create_post_header': "Create a New Post", 'post_title_label': "Title", 'post_title_placeholder': "Enter a clear and concise title", 'post_content_label': "Your Question or Tip", 'post_content_placeholder': "Describe your issue or share your knowledge in detail...",
        'upload_image_label': "Upload an Image (Optional)", 'submit_post_button': "Submit Post", 'post_success_message': "✅ Your post has been published!",
        'all_posts_header': "All Posts", 'posted_by': "Posted by {username} on {date}",
        'like_button': "👍 Like ({count})", 'reply_button': "💬 Reply", 'replies_header': "Replies", 'add_reply_placeholder': "Write a reply...", 'submit_reply_button': "Post Reply",
        'no_posts_message': "No posts yet. Be the first to start a conversation!", 'view_replies': "View {count} replies", 'hide_replies': "Hide replies",
        'soil_characteristics_heading': "🔬 Characteristics", 'soil_suitable_crops_heading': "🌱 Suitable Crops", 'soil_model_not_found_warning': "⚠ Soil prediction model not found. Please contact the administrator.",
        'set_reminder_button': "Set Sowing Reminder",
        'reminder_set_success': "✅ Reminder set! An SMS with the sowing dates has been sent to your registered number.",
        'reminder_set_failure': "❌ Could not set reminder. Please ensure your contact number is updated in your profile.",
        'stage_seedling': 'Seedling', 'stage_vegetative': 'Vegetative', 'stage_flowering': 'Flowering', 'stage_fruiting': 'Fruiting', 'stage_maturity': 'Maturity',
        'column_days_range': 'Days Range', 'column_fertilizer': 'Fertilizer', 'column_dosage': 'Dosage (kg/acre)', 'column_frequency': 'Frequency', 'column_irrigation_depth': 'Irrigation Depth / Volume', 'column_irrigation_frequency': 'Irrigation Frequency',
    },
    'hi': {
        'app_title': "कृषि-सखी", 'go_to': "पेज पर जाएं", 'go_button': "जाएं", 'back_to_home': "होम पर वापस जाएं", 'language_selector_label': "भाषा",
        'page_home': "होम", 'page_crop_recommendation': "फसल सिफारिश", 'page_yield_prediction': "उपज भविष्यवाणी", 'page_crop_calendar': "फसल कैलेंडर", 'page_weather_report': "मौसम रिपोर्ट", 'page_leaf_disease_prediction': "पत्ती रोग की भविष्यवाणी", 'page_translator': "अनुवादक", 'page_profile': "प्रोफ़ाइल", 'page_logs': "लॉग", 'page_community_forum': "सामुदायिक मंच",
        'quick_actions_title': "🚀 त्वरित क्रियाएँ", 'quick_action_crop_desc': "अपनी मिट्टी और मौसम के आधार पर उपयुक्त फसल सुझाव प्राप्त करें।", 'quick_action_weather_desc': "अपने क्षेत्र के मौसम की जांच करें और अपनी कृषि गतिविधियों की योजना बनाएं।",
        'page_crop_guide': "फसल गाइड", 'page_pest_prediction': "कीट भविष्यवाणी", 'page_soil_type_prediction': "मिट्टी का प्रकार भविष्यवाणी",
        'crop_guide_title': "फसल फर्टिगेशन और सिंचाई गाइड", 'select_stage_label': "एक विकास चरण चुनें", 'guide_table_header': "मार्गदर्शन विवरण",
        'file_not_found_error': "फ़ाइल नहीं मिली। कृपया सुनिश्चित करें कि यह सही फ़ोल्डर में है।",
        'home_title': "कृषि-सखी में आपका स्वागत है! 🌱", 'home_subtitle': "आधुनिक खेती के लिए आपका एआई-संचालित सहायक।",
        'home_intro': "*कृषि-सखी* को किसानों को डेटा-संचालित अंतर्दृष्टि के साथ सशक्त बनाने के लिए डिज़ाइन किया गया है।\nविभिन्न सुविधाओं तक पहुंचने के लिए साइडबार में अनुभागों के माध्यम से नेविगेट करें:\n- *🌾 फसल सिफारिश:* मिट्टी और पर्यावरणीय कारकों के आधार पर लगाने के लिए सर्वोत्तम फसलों के लिए सुझाव प्राप्त करें।\n- *📈 उपज भविष्यवाणी:* ऐतिहासिक डेटा और खेती के मापदंडों के आधार पर अपनी फसल की उपज का पूर्वानुमान करें।\n- *🗓 फसल कैलेंडर:* केरल में विभिन्न फसलों के लिए आदर्श बुवाई और कटाई का समय देखें।\n- *🌦 मौसम रिपोर्ट:* अपने स्थान के लिए वास्तविक समय के मौसम अपडेट और पूर्वानुमान प्राप्त करें।\n- *🌿 पत्ती रोग की भविष्यवाणी:* बीमारियों का शीघ्र पता लगाने के लिए पत्ती की एक छवि अपलोड करें।\n- *🌐 अनुवादक:* खेती की शर्तों और वाक्यों का विभिन्न भारतीय भाषाओं के बीच अनुवाद करें।\n- *👥 सामुदायिक मंच:* अन्य किसानों से जुड़ें, प्रश्न पूछें और समाधान साझा करें।",
        'crop_rec_header': "🌾 फसल सिफारिश", 'crop_rec_intro': "फसल की सिफारिश प्राप्त करने के लिए निम्नलिखित विवरण दर्ज करें।",
        'nitrogen_label': "नाइट्रोजन (N)", 'phosphorus_label': "फास्फोरस (P)", 'potassium_label': "पोटेशियम (K)", 'temperature_label': "तापमान (°C)", 'humidity_label': "नमी (%)", 'ph_label': "पीएच मान", 'rainfall_label': "वर्षा (मिमी)",
        'recommend_crop_button': "फसल की सिफारिश करें", 'recommendation_success': "🎉 अनुशंसित फसल है *{crop}*।",
        'crop_info_header': "{crop} के बारे में अधिक जानकारी", 'description_heading': "📖 विवरण", 'irrigation_heading': "💧 सिंचाई युक्तियाँ", 'conditions_heading': "☀️ आदर्श स्थितियाँ", 'pests_heading': "🐛 सामान्य कीट और रोग",
        'yield_pred_header': "📈 उपज भविष्यवाणी", 'yield_pred_intro': "केरल के एक विशिष्ट क्षेत्र के लिए फसल की उपज की भविष्यवाणी करने के लिए नीचे दिए गए विवरण भरें।",
        'year_label': "वर्ष", 'crop_label': "फसल", 'season_label': "मौसम", 'area_label': "क्षेत्र (हेक्टेयर में)",
        'predict_yield_button': "उपज की भविष्यवाणी करें", 'yield_prediction_success': "🌾 अनुमानित फसल उपज लगभग *{yield_val:.2f} किग्रा/हेक्टेयर* है।",
        'market_price_header': "💰 अनुमानित बाजार मूल्य", 'fetching_prices_spinner': "नवीनतम बाजार मूल्य प्राप्त हो रहे हैं...", 'market_price_label': "औसत बाजार मूल्य (प्रति क्विंटल)", 'total_value_label': "कुल अनुमानित मूल्य", 'profit_header': "💵 अनुमानित लाभ", 'cost_per_ha_label': "खेती लागत प्रति हेक्टेयर (₹)", 'total_cost_label': "कुल खेती लागत", 'total_profit_label': "कुल अनुमानित लाभ", 'price_not_available': "{crop} के लिए मूल्य डेटा वर्तमान में अनुपलब्ध है।", 'price_api_error': "इस समय बाजार मूल्य डेटा प्राप्त नहीं किया जा सका।",
        'crop_calendar_header': "🗓 फसल बुवाई और कटाई कैलेंडर", 'select_crop_label': "एक फसल चुनें",
        'schedule_for_crop': "📅 {crop} के लिए अनुसूची", 'sowing_start_label': "बुवाई शुरू", 'sowing_end_label': "बुवाई अंत", 'harvest_start_label': "कटाई शुरू", 'harvest_end_label': "कटाई अंत", 'fertiliser_label': "उर्वरक (N:P:K किग्रा/हेक्टेयर)", 'duration_label': "अवधि",
        'weather_report_header': "🌦 मौसम रिपोर्ट", 'enter_city_label': "अपने शहर का नाम दर्ज करें", 'get_weather_button': "मौसम प्राप्त करें",
        'current_weather_in': "{location} में वर्तमान मौसम", 'temperature_metric': "तापमान", 'feels_like_metric': "महसूस होता है", 'humidity_metric': "नमी", 'description_label': "विवरण:", 'wind_label': "हवा:", 'precipitation_label': "वर्षा:",
        'forecast_header': "🗓 3-दिवसीय पूर्वानुमान", 'avg_temp_label': "औसत तापमान:", 'max_temp_label': "अधिकतम तापमान:", 'min_temp_label': "न्यूनतम तापमान:", 'sunrise_label': "सूर्योदय:", 'sunset_label': "सूर्यास्त:", 'outlook_label': "दृष्टिकोण:",
        'error_parsing_weather': "मौसम डेटा पार्स नहीं किया जा सका। कृपया शहर का नाम जांचें और पुनः प्रयास करें। त्रुटि: {e}", 'enter_city_warning': "⚠ कृपया एक शहर का नाम दर्ज करें।",
        'translator_header': "🌐 किसान भाषा अनुवादक", 'translator_intro': "✍ अनुवाद करने के लिए टेक्स्ट दर्ज करें", 'translate_button': "🔄 अनुवाद करें",
        'translation_failed_error': "❌ अनुवाद विफल: {e}", 'enter_text_warning': "⚠ कृपया कुछ टेक्स्ट दर्ज करें।",
        'ai_chatbot_header': "🤖 एआई चैटबॉट", 'ai_chatbot_intro': "यह सुविधा जल्द ही आ रही है! अपनी खेती के सवालों में मदद के लिए एक शक्तिशाली एआई चैटबॉट के लिए बने रहें।",
        'languages_supported_label': "समर्थित भाषाएँ", 'farmers_helped_label': "किसानों की मदद की", 'crops_covered_label': "शामिल फसलें", 'recommendations_given_label': "एआई सिफारिशें",
        'leaf_disease_header': "🌿 पत्ती रोग की भविष्यवाणी", 'leaf_disease_intro': "बीमारियों की जांच के लिए पौधे की पत्ती का चित्र अपलोड करें।", 'model_not_found_warning': "⚠ रोग भविष्यवाणी मॉडल नहीं मिला। कृपया व्यवस्थापक से संपर्क करें।",
        'uploader_label': "एक पत्ती की छवि चुनें...", 'uploaded_image_caption': "अपलोड की गई छवि।", 'predict_disease_button': "🔍 रोग की भविष्यवाणी करें",
        'predicting_spinner': "पत्ती का विश्लेषण किया जा रहा है...", 'prediction_result': "✅ भविष्यवाणी: *{disease}*",
        'remedy_info_header': "{disease} के लिए उपचार और प्रबंधन", 'disease_description_heading': "ℹ️ रोग के बारे में", 'organic_remedies_heading': "🌿 जैविक उपचार", 'chemical_remedies_heading': "🧪 रासायनिक उपचार", 'healthy_plant_message': "खुशखबरी! आपका पौधा स्वस्थ दिखाई देता है। अच्छी कृषि पद्धतियों को जारी रखें।",
        'pest_prediction_header': "छवि से कीट की पहचान", 'upload_image_pest': "कीट या प्रभावित पत्ती की एक छवि अपलोड करें", 'prediction_pest': "अनुमानित कीट", 'model_loading_error': "कीट भविष्यवाणी मॉडल लोड नहीं हो सका। कृपया इंटरनेट कनेक्शन की जांच करें।",
        'remedy_info_header_pest': "{pest} के लिए उपचार और प्रबंधन", 'pest_description_heading': "ℹ️ कीट के बारे में", 'no_remedy_info': "{pest} के लिए उपचार की जानकारी अभी उपलब्ध नहीं है।",
        'feedback_header': "प्रतिक्रिया", 'rating_label': "अपने अनुभव का मूल्यांकन करें:", 'submit_button': "प्रस्तुत करें", 'feedback_success': "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        'contact_header': "संपर्क करें",
        'weather_api_key_error': "API कुंजी त्रुटि: {message}। कृपया अपनी OpenWeatherMap API कुंजी secrets.toml में जांचें। ध्यान दें कि नई कुंजियों को सक्रिय होने में कुछ घंटे लग सकते हैं।", 'weather_city_not_found_error': "शहर नहीं मिला त्रुटि: {message}। कृपया शहर के नाम की वर्तनी जांचें।", 'weather_api_generic_error': "API त्रुटि ({cod}): {message}",
        'login_page_title': "कृषि-सखी में आपका स्वागत है 🌱", 'login_tab': "लॉग इन करें", 'signup_tab': "साइन अप करें",
        'username_label': "उपयोगकर्ता नाम", 'password_label': "पासवर्ड", 'login_button': "लॉग इन करें", 'signup_button': "साइन अप करें", 'logout_button': "लॉग आउट",
        'login_error': "गलत उपयोगकर्ता नाम या पासवर्ड।", 'signup_success': "पंजीकरण सफल! कृपया लॉग इन करें।", 'signup_error_exists': "उपयोगकर्ता नाम पहले से मौजूद है। कृपया दूसरा चुनें।", 'signup_error_general': "पंजीकरण के दौरान एक त्रुटि हुई।",
        'welcome_back': "वापसी पर स्वागत है, {username}!",
        'profile_header': "👤 मेरी प्रोफ़ाइल", 'profile_subheader': "अपने खाते का विवरण और सेटिंग्स प्रबंधित करें।",
        'full_name_label': "पूरा नाम", 'contact_number_label': "संपर्क नंबर", 'new_username_label': "नया उपयोगकर्ता नाम", 'new_password_label': "नया पासवर्ड", 'confirm_password_label': "नए पासवर्ड की पुष्टि करें", 'profile_pic_label': "प्रोफ़ाइल तस्वीर",
        'update_profile_button': "प्रोफ़ाइल अपडेट करें", 'profile_update_success': "✅ प्रोफ़ाइल सफलतापूर्वक अपडेट की गई!", 'profile_update_error': "❌ प्रोफ़ाइल अपडेट नहीं हो सकी। उपयोगकर्ता नाम पहले से मौजूद हो सकता है।", 'password_mismatch_error': "पासवर्ड मेल नहीं खाते।",
        'danger_zone_header': "🚨 खतरा क्षेत्र", 'delete_account_button': "मेरा खाता हटाएं", 'delete_account_warning': "यह कार्रवाई अपरिवर्तनीय है। आपका सारा डेटा स्थायी रूप से हटा दिया जाएगा।", 'delete_confirmation_label': "पुष्टि करने के लिए, कृपया नीचे दिए गए बॉक्स में `DELETE` टाइप करें।",
        'delete_account_confirm_button': "हाँ, मैं अपना खाता हटाना चाहता हूँ", 'account_deleted_success': "आपका खाता सफलतापूर्वक हटा दिया गया है।", 'incorrect_delete_confirmation': "गलत पुष्टि पाठ।",
        'logs_header': "मेरे खेती लॉग", 'logs_intro': "अपनी दैनिक गतिविधियों, अवलोकनों और नोट्स को यहाँ रिकॉर्ड करें।", 'new_log_label': "नई लॉग प्रविष्टि", 'save_log_button': "लॉग सहेजें", 'log_saved_success': "लॉग प्रविष्टि सफलतापूर्वक सहेजी गई!",
        'past_logs_header': "📖 पिछली प्रविष्टियाँ", 'no_logs_message': "आपके पास अभी तक कोई लॉग प्रविष्टि नहीं है।", 'delete_log_button': "हटाएं", 'log_deleted_success': "लॉग हटाया गया!", 'record_log_prompt': "एक वॉयस नोट संलग्न करें:",
        'voice_note_recorded': "🎤 वॉयस नोट रिकॉर्ड किया गया और आपके लॉग के साथ सहेजने के लिए तैयार है।", 'attached_voice_note': "संलग्न वॉयस नोट:", 'upload_media_label': "तस्वीरें या वीडियो संलग्न करें:",
        'attached_images': "संलग्न तस्वीरें:", 'attached_videos': "संलग्न वीडियो:",
        'forum_header': "👥 सामुदायिक मंच", 'forum_intro': "अन्य किसानों से जुड़ें। प्रश्न पूछें, समाधान साझा करें और मिलकर आगे बढ़ें।",
        'create_post_header': "एक नई पोस्ट बनाएं", 'post_title_label': "शीर्षक", 'post_title_placeholder': "एक स्पष्ट और संक्षिप्त शीर्षक दर्ज करें", 'post_content_label': "आपका प्रश्न या सुझाव", 'post_content_placeholder': "अपनी समस्या का वर्णन करें या अपना ज्ञान विस्तार से साझा करें...",
        'upload_image_label': "एक छवि अपलोड करें (वैकल्पिक)", 'submit_post_button': "पोस्ट सबमिट करें", 'post_success_message': "✅ आपकी पोस्ट प्रकाशित हो गई है!",
        'all_posts_header': "सभी पोस्ट", 'posted_by': "{username} द्वारा {date} को पोस्ट किया गया", 'like_button': "👍 पसंद करें ({count})", 'reply_button': "💬 उत्तर दें", 'replies_header': "उत्तर", 'add_reply_placeholder': "एक उत्तर लिखें...",
        'submit_reply_button': "उत्तर पोस्ट करें", 'no_posts_message': "अभी तक कोई पोस्ट नहीं है। बातचीत शुरू करने वाले पहले व्यक्ति बनें!", 'view_replies': "{count} उत्तर देखें", 'hide_replies': "उत्तर छिपाएं",
        'soil_prediction_header': "🏞️ मिट्टी का प्रकार भविष्यवाणी", 'soil_prediction_intro': "मिट्टी के प्रकार की पहचान करने और उसके गुणों के बारे में जानने के लिए मिट्टी की एक छवि अपलोड करें।", 'upload_soil_image_label': "एक मिट्टी की छवि चुनें...",
        'predict_soil_button': "🔍 मिट्टी के प्रकार की भविष्यवाणी करें", 'analyzing_soil_spinner': "मिट्टी का विश्लेषण किया जा रहा है...", 'soil_prediction_result': "✅ अनुमानित मिट्टी का प्रकार है: *{soil_type}*", 'soil_info_header': "{soil_type} के बारे में जानकारी",
        'soil_characteristics_heading': "🔬 विशेषताएँ", 'soil_suitable_crops_heading': "🌱 उपयुक्त फसलें", 'soil_model_not_found_warning': "⚠ मिट्टी भविष्यवाणी मॉडल नहीं मिला। कृपया व्यवस्थापक से संपर्क करें।",
        'set_reminder_button': "बुवाई अनुस्मारक सेट करें",
        'reminder_set_success': "✅ अनुस्मारक सेट! बुवाई की तारीखों के साथ एक एसएमएस आपके पंजीकृत नंबर पर भेजा गया है।",
        'reminder_set_failure': "❌ अनुस्मारक सेट नहीं किया जा सका। कृपया सुनिश्चित करें कि आपकी प्रोफ़ाइल में आपका संपर्क नंबर अपडेट है।",
        'stage_seedling': 'अंकुरण', 'stage_vegetative': 'वनस्पति', 'stage_flowering': 'फूलन', 'stage_fruiting': 'फलन', 'stage_maturity': 'परिपक्वता',
        'column_days_range': 'दिनों की सीमा', 'column_fertilizer': 'उर्वरक', 'column_dosage': 'खुराक (किग्रा/एकड़)', 'column_frequency': 'आवृत्ति', 'column_irrigation_depth': 'सिंचाई गहराई / मात्रा', 'column_irrigation_frequency': 'सिंचाई आवृत्ति',
    },
    'ml': {
        'app_title': "കൃഷിസഖി", 'go_to': "പോകുക", 'go_button': "പോകൂ", 'back_to_home': "വീട്ടിലേക്കു തിരിച്ചു പോകുക", 'language_selector_label': "ഭാഷ",
        'page_home': "ഹോം", 'page_crop_recommendation': "വിള ശുപാർശ", 'page_yield_prediction': "വിളവ് പ്രവചനം", 'page_crop_calendar': "വിള കലണ്ടർ", 'page_weather_report': "കാലാവസ്ഥാ റിപ്പോർട്ട്", 'page_leaf_disease_prediction': "ഇലരോഗ നിർണ്ണയം", 'page_translator': "പരിഭാഷകൻ", 'page_profile': "പ്രൊഫൈൽ", 'page_logs': "ലോഗുകൾ", 'page_community_forum': "കമ്മ്യൂണിറ്റി ഫോറം",
        'quick_actions_title': "🚀 ക്വിക്ക് ആക്ഷനുകൾ", 'quick_action_crop_desc': "നിങ്ങളുടെ മണ്ണും കാലാവസ്ഥയും അടിസ്ഥാനമാക്കി അനുയോജ്യമായ വിള നിർദ്ദേശങ്ങൾ ലഭിക്കുക.", 'quick_action_weather_desc': "നിങ്ങളുടെ പ്രദേശത്തെ കാലാവസ്ഥാ പ്രവണത പരിശോധിച്ച് നിങ്ങളുടെ കൃഷി പ്രവർത്തനങ്ങൾ திட்டമിടുക.",
        'page_crop_guide': "വിള ഗൈഡ്", 'page_pest_prediction': "കീട പ്രവചനം", 'page_soil_type_prediction': "മണ്ണിൻ്റെ തരം പ്രവചനം",
        'crop_guide_title': "വിള വളപ്രയോഗ-ജലസേചന ഗൈഡ്", 'select_stage_label': "വളർച്ചാ ഘട്ടം തിരഞ്ഞെടുക്കുക", 'guide_table_header': "മാർഗ്ഗനിർദ്ദേശങ്ങൾ",
        'file_not_found_error': "ഫയൽ കണ്ടെത്തിയില്ല. ദയവായി അത് ശരിയായ ഫോൾഡറിൽ ആണെന്ന് ഉറപ്പാക്കുക.",
        'home_title': "കൃഷിസഖിയിലേക്ക് സ്വാഗതം! 🌱", 'home_subtitle': "ആധുനിക കൃഷിക്കായുള്ള നിങ്ങളുടെ AI-പവർ അസിസ്റ്റന്റ്.",
        'home_intro': "*കൃഷിസഖി* കർഷകർക്ക് ഡാറ്റാധിഷ്ഠിത ഉൾക്കാഴ്ചകൾ നൽകാൻ രൂപകൽപ്പന ചെയ്തിട്ടുള്ളതാണ്.\nവിവിധ ഫീച്ചറുകൾ ആക്‌സസ് ചെയ്യുന്നതിന് സൈഡ്‌ബാറിലെ വിഭാഗങ്ങളിലൂടെ നാവിഗേറ്റ് ചെയ്യുക:\n- *🌾 വിള ശുപാർശ:* മണ്ണും പാരിസ്ഥിതിക ഘടകങ്ങളും അടിസ്ഥാനമാക്കി നടാൻ ഏറ്റവും മികച്ച വിളകൾക്കുള്ള നിർദ്ദേശങ്ങൾ നേടുക.\n- *📈 വിളവ് പ്രവചനം:* ചരിത്രപരമായ ഡാറ്റയും കൃഷി പാരാമീറ്ററുകളും അടിസ്ഥാനമാക്കി നിങ്ങളുടെ വിളവ് പ്രവചിക്കുക.\n- *🗓 വിള കലണ്ടർ:* കേരളത്തിലെ വിവിധ വിളകളുടെ അനുയോജ്യമായ വിതയ്ക്കൽ, വിളവെടുപ്പ് സമയം കാണുക.\n- *🌦 കാലാവസ്ഥാ റിപ്പോർട്ട്:* നിങ്ങളുടെ സ്ഥലത്തിനായുള്ള തത്സമയ കാലാവസ്ഥാ അപ്‌ഡേറ്റുകളും പ്രവചനങ്ങളും നേടുക.\n- *🌿 ഇലരോഗ നിർണ്ണയം:* രോഗങ്ങൾ നേരത്തെ കണ്ടെത്താൻ ഇലയുടെ ഒരു ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.\n- *🌐 പരിഭാഷകൻ:* കൃഷി സംബന്ധമായ പദങ്ങളും വാക്യങ്ങളും വിവിധ ഇന്ത്യൻ ഭാഷകളിലേക്ക് വിവർത്തനം ചെയ്യുക.\n- *👥 കമ്മ്യൂണിറ്റി ഫോറം:* മറ്റ് കർഷകരുമായി ബന്ധപ്പെടുക, ചോദ്യങ്ങൾ ചോദിക്കുക, പരിഹാരങ്ങൾ പങ്കിടുക.",
        'crop_rec_header': "🌾 വിള ശുപാർശ", 'crop_rec_intro': "വിള ശുപാർശ ലഭിക്കുന്നതിന് താഴെ പറയുന്ന വിവരങ്ങൾ നൽകുക.",
        'nitrogen_label': "നൈട്രജൻ (N)", 'phosphorus_label': "ഫോസ്ഫറസ് (P)", 'potassium_label': "പൊട്ടാസ്യം (K)", 'temperature_label': "താപനില (°C)", 'humidity_label': "ഈർപ്പം (%)", 'ph_label': "pH മൂല്യം", 'rainfall_label': "മഴ (mm)",
        'recommend_crop_button': "വിള ശുപാർശ ചെയ്യുക", 'recommendation_success': "🎉 ശുപാർശ ചെയ്യുന്ന വിള *{crop}* ആണ്.",
        'crop_info_header': "{crop}-നെക്കുറിച്ച് കൂടുതൽ വിവരങ്ങൾ", 'description_heading': "📖 വിവരണം", 'irrigation_heading': "💧 ജലസേചന നുറുങ്ങുകൾ", 'conditions_heading': "☀️ അനുയോജ്യമായ സാഹചര്യങ്ങൾ", 'pests_heading': "🐛 സാധാരണ കീടങ്ങളും രോഗങ്ങളും",
        'yield_pred_header': "📈 വിളവ് പ്രവചനം", 'yield_pred_intro': "കേരളത്തിലെ ഒരു പ്രത്യേക പ്രദേശത്തെ വിളവ് പ്രവചിക്കാൻ താഴെയുള്ള വിശദാംശങ്ങൾ പൂരിപ്പിക്കുക.",
        'year_label': "വർഷം", 'crop_label': "വിള", 'season_label': "സീസൺ", 'area_label': "വിസ്തീർണ്ണം (ഹെക്ടറിൽ)",
        'predict_yield_button': "വിളവ് പ്രവചിക്കുക", 'yield_prediction_success': "🌾 പ്രവചിക്കപ്പെട്ട വിളവ് ഏകദേശം *{yield_val:.2f} കി.ഗ്രാം/ഹെക്ടർ* ആണ്.",
        'market_price_header': "💰 കണക്കാക്കിയ വിപണി മൂല്യം", 'fetching_prices_spinner': "ഏറ്റവും പുതിയ മാർക്കറ്റ് വിലകൾ ലഭ്യമാക്കുന്നു...", 'market_price_label': "ശരാശരി വിപണി വില (ക്വിൻ്റലിന്)", 'total_value_label': "കണക്കാക്കിയ മൊത്തം മൂല്യം", 'profit_header': "💵 കണക്കാക്കിയ ലാഭം", 'cost_per_ha_label': "കൃഷി ചെലവ് ഹെക്ടറിന് (₹)", 'total_cost_label': "മൊത്തം കൃഷി ചെലവ്", 'total_profit_label': "കണക്കാക്കിയ മൊത്തം ലാഭം", 'price_not_available': "{crop}-നുള്ള വിലവിവരം ഇപ്പോൾ ലഭ്യമല്ല.", 'price_api_error': "മാർക്കറ്റ് വില വിവരങ്ങൾ ഇപ്പോൾ ലഭ്യമാക്കാൻ സാധിക്കുന്നില്ല.",
        'crop_calendar_header': "🗓 വിള വിതയ്ക്കൽ, വിളവെടുപ്പ് കലണ്ടർ", 'select_crop_label': "ഒരു വിള തിരഞ്ഞെടുക്കുക",
        'schedule_for_crop': "📅 {crop}-നുള്ള ഷെഡ്യൂൾ", 'sowing_start_label': "വിതയ്ക്കൽ ആരംഭം", 'sowing_end_label': "വിതയ്ക്കൽ അവസാനം", 'harvest_start_label': "വിളവെടുപ്പ് ആരംഭം", 'harvest_end_label': "വിളവെടുപ്പ് അവസാനം", 'fertiliser_label': "വളം (N:P:K കി.ഗ്രാം/ഹെക്ടർ)", 'duration_label': "കാലാവധി",
        'weather_report_header': "🌦 കാലാവസ്ഥാ റിപ്പോർട്ട്", 'enter_city_label': "നിങ്ങളുടെ നഗരത്തിന്റെ പേര് നൽകുക", 'get_weather_button': "കാലാവസ്ഥ നേടുക",
        'current_weather_in': "{location}-ലെ നിലവിലെ കാലാവസ്ഥ", 'temperature_metric': "താപനില", 'feels_like_metric': "അനുഭവപ്പെടുന്നത്", 'humidity_metric': "ഈർപ്പം",
        'description_label': "വിവരണം:", 'wind_label': "കാറ്റ്:", 'precipitation_label': "മഴ:",
        'forecast_header': "🗓 3-ദിവസത്തെ പ്രവചനം", 'avg_temp_label': "ശരാശരി താപനില:", 'max_temp_label': "പരമാവധി താപനില:", 'min_temp_label': "കുറഞ്ഞ താപനില:", 'sunrise_label': "സൂര്യോദയം:", 'sunset_label': "സൂര്യാസ്തമയം:", 'outlook_label': "കാഴ്ചപ്പാട്:",
        'error_parsing_weather': "കാലാവസ്ഥാ ഡാറ്റ പാഴ്‌സ് ചെയ്യാൻ കഴിഞ്ഞില്ല. ദയവായി നഗരത്തിന്റെ പേര് പരിശോധിച്ച് വീണ്ടും ശ്രമിക്കുക. പിശക്: {e}", 'enter_city_warning': "⚠ ദയവായി ഒരു നഗരത്തിന്റെ പേര് നൽകുക.",
        'translator_header': "🌐 കർഷക ഭാഷാ പരിഭാഷകൻ", 'translator_intro': "✍ വിവർത്തനം ചെയ്യാൻ ടെക്സ്റ്റ് നൽകുക", 'translate_button': "🔄 വിവർത്തനം ചെയ്യുക",
        'translation_failed_error': "❌ വിവർത്തനം പരാജയപ്പെട്ടു: {e}", 'enter_text_warning': "⚠ ദയവായി കുറച്ച് ടെക്സ്റ്റ് നൽകുക.",
        'ai_chatbot_header': "🤖 AI ചാറ്റ്ബോട്ട്", 'ai_chatbot_intro': "ഈ ഫീച്ചർ ഉടൻ വരുന്നു! നിങ്ങളുടെ കൃഷി ചോദ്യങ്ങൾക്ക് സഹായിക്കാൻ ഒരു ശക്തമായ AI ചാറ്റ്ബോട്ടിനായി കാത്തിരിക്കുക.",
        'languages_supported_label': "പിന്തുണയ്ക്കുന്ന ഭാഷകൾ", 'farmers_helped_label': "സഹായിച്ച കർഷകർ", 'crops_covered_label': "ഉൾക്കൊള്ളുന്ന വിളകൾ", 'recommendations_given_label': "AI ശുപാർശകൾ",
        'leaf_disease_header': "🌿 ഇലരോഗ നിർണ്ണയം", 'leaf_disease_intro': "രോഗങ്ങൾ പരിശോധിക്കാൻ സസ്യത്തിന്റെ ഇലയുടെ ഒരു ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.", 'model_not_found_warning': "⚠ രോഗ നിർണ്ണയ മോഡൽ കണ്ടെത്തിയില്ല. ദയവായി അഡ്മിനിസ്ട്രേറ്ററുമായി ബന്ധപ്പെടുക.",
        'uploader_label': "ഇലയുടെ ചിത്രം തിരഞ്ഞെടുക്കുക...", 'uploaded_image_caption': "അപ്‌ലോഡ് ചെയ്ത ചിത്രം.", 'predict_disease_button': "🔍 രോഗം നിർണ്ണയിക്കുക",
        'predicting_spinner': "ഇല വിശകലനം ചെയ്യുന്നു...", 'prediction_result': "✅ നിർണ്ണയം: *{disease}*",
        'remedy_info_header': "{disease}-നുള്ള ചികിത്സയും പ്രതിരോധവും", 'disease_description_heading': "ℹ️ രോഗത്തെക്കുറിച്ച്", 'organic_remedies_heading': "🌿 ജൈവ പ്രതിവിധികൾ", 'chemical_remedies_heading': "🧪 രാസപരമായ പ്രതിവിധികൾ", 'healthy_plant_message': "ഒരു സന്തോഷ വാർത്ത! നിങ്ങളുടെ ചെടി ആരോഗ്യത്തോടെ കാണപ്പെടുന്നു. നല്ല കൃഷിരീതികൾ തുടരുക.",
        'soil_prediction_header': "🏞️ മണ്ണിൻ്റെ തരം പ്രവചനം", 'soil_prediction_intro': "മണ്ണിൻ്റെ തരം തിരിച്ചറിയാനും അതിൻ്റെ ഗുണങ്ങളെക്കുറിച്ച് പഠിക്കാനും മണ്ണിൻ്റെ ഒരു ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.", 'upload_soil_image_label': "മണ്ണിൻ്റെ ഒരു ചിത്രം തിരഞ്ഞെടുക്കുക...",
        'predict_soil_button': "🔍 മണ്ണിൻ്റെ തരം പ്രവചിക്കുക", 'analyzing_soil_spinner': "മണ്ണ് വിശകലനം ചെയ്യുന്നു...", 'soil_prediction_result': "✅ പ്രവചിച്ച മണ്ണിൻ്റെ തരം: *{soil_type}*", 'soil_info_header': "{soil_type}-നെക്കുറിച്ചുള്ള വിവരങ്ങൾ",
        'pest_prediction_header': "ചിത്രത്തിൽ നിന്ന് കീടങ്ങളെ തിരിച്ചറിയൽ", 'upload_image_pest': "കീടത്തിന്റെയോ രോഗം ബാധിച്ച ഇലയുടെയോ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക", 'prediction_pest': "പ്രവചിച്ച കീടം", 'model_loading_error': "കീട പ്രവചന മോഡൽ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല. ദയവായി ഇന്റർനെറ്റ് കണക്ഷൻ പരിശോധിക്കുക.",
        'remedy_info_header_pest': "{pest}-നുള്ള ചികിത്സയും പ്രതിരോധവും", 'pest_description_heading': "ℹ️ കീടത്തെക്കുറിച്ച്", 'no_remedy_info': "{pest}-നുള്ള പ്രതിവിധി വിവരങ്ങൾ ലഭ്യമല്ല.",
        'feedback_header': "അഭിപ്രായം", 'rating_label': "നിങ്ങളുടെ അനുഭവം റേറ്റുചെയ്യുക:", 'submit_button': "സമർപ്പിക്കുക", 'feedback_success': "നിങ്ങളുടെ അഭിപ്രായത്തിന് നന്ദി!",
        'contact_header': "ബന്ധപ്പെടുക",
        'weather_api_key_error': "API കീ പിശക്: {message}. secrets.toml-ൽ നിങ്ങളുടെ OpenWeatherMap API കീ പരിശോധിക്കുക. പുതിയ കീകൾ സജീവമാകാൻ കുറച്ച് മണിക്കൂറുകൾ എടുത്തേക്കാം.", 'weather_city_not_found_error': "നഗരം കണ്ടെത്താനായില്ല പിശക്: {message}. ദയവായി നഗരത്തിന്റെ പേരിന്റെ സ്പെല്ലിംഗ് പരിശോധിക്കുക.", 'weather_api_generic_error': "API പിശക് ({cod}): {message}",
        'login_page_title': "കൃഷിസഖിയിലേക്ക് സ്വാഗതം 🌱", 'login_tab': "ലോഗിൻ ചെയ്യുക", 'signup_tab': "സൈൻ അപ്പ് ചെയ്യുക",
        'username_label': "ഉപയോക്തൃനാമം", 'password_label': "പാസ്വേഡ്", 'login_button': "ലോഗിൻ ചെയ്യുക", 'signup_button': "സൈൻ അപ്പ് ചെയ്യുക", 'logout_button': "ലോഗ് ഔട്ട്",
        'login_error': "തെറ്റായ ഉപയോക്തൃനാമം അല്ലെങ്കിൽ പാസ്‌വേഡ്.", 'signup_success': "രജിസ്ട്രേഷൻ വിജയകരം! ദയവായി ലോഗിൻ ചെയ്യുക.", 'signup_error_exists': "ഉപയോക്തൃനാമം നിലവിലുണ്ട്. ദയവായി മറ്റൊന്ന് തിരഞ്ഞെടുക്കുക.", 'signup_error_general': "രജിസ്ട്രേഷൻ സമയത്ത് ഒരു പിശക് സംഭവിച്ചു.",
        'welcome_back': "തിരികെ സ്വാഗതം, {username}!",
        'profile_header': "👤 എൻ്റെ പ്രൊഫൈൽ", 'profile_subheader': "നിങ്ങളുടെ അക്കൗണ്ട് വിശദാംശങ്ങളും ക്രമീകരണങ്ങളും നിയന്ത്രിക്കുക.",
        'full_name_label': "മുഴുവൻ പേര്", 'contact_number_label': "ബന്ധപ്പെടാനുള്ള നമ്പർ", 'new_username_label': "പുതിയ ഉപയോക്തൃനാമം", 'new_password_label': "പുതിയ പാസ്‌വേഡ്", 'confirm_password_label': "പുതിയ പാസ്‌വേഡ് സ്ഥിരീകരിക്കുക", 'profile_pic_label': "പ്രൊഫൈൽ ചിത്രം",
        'update_profile_button': "പ്രൊഫൈൽ അപ്ഡേറ്റ് ചെയ്യുക", 'profile_update_success': "✅ പ്രൊഫൈൽ വിജയകരമായി അപ്ഡേറ്റ് ചെയ്തു!", 'profile_update_error': "❌ പ്രൊഫൈൽ അപ്ഡേറ്റ് ചെയ്യാൻ കഴിഞ്ഞില്ല. ഉപയോക്തൃനാമം ഇതിനകം നിലവിലുണ്ടാകാം.", 'password_mismatch_error': "പാസ്‌വേഡുകൾ പൊരുത്തപ്പെടുന്നില്ല.",
        'danger_zone_header': "🚨 അപകട മേഖല", 'delete_account_button': "എൻ്റെ അക്കൗണ്ട് ഇല്ലാതാക്കുക", 'delete_account_warning': "ഈ പ്രവർത്തനം മാറ്റാനാവാത്തതാണ്. നിങ്ങളുടെ എല്ലാ ഡാറ്റയും ശാശ്വതമായി ഇല്ലാതാക്കപ്പെടും.", 'delete_confirmation_label': "സ്ഥിരീകരിക്കുന്നതിന്, ദയവായി താഴെയുള്ള ബോക്സിൽ `DELETE` എന്ന് ടൈപ്പ് ചെയ്യുക.",
        'delete_account_confirm_button': "അതെ, എൻ്റെ അക്കൗണ്ട് ഇല്ലാതാക്കാൻ ഞാൻ ആഗ്രഹിക്കുന്നു", 'account_deleted_success': "നിങ്ങളുടെ അക്കൗണ്ട് വിജയകരമായി ഇല്ലാതാക്കി.", 'incorrect_delete_confirmation': "തെറ്റായ സ്ഥിരീകരണ വാചകം.",
        'logs_header': "എൻ്റെ കൃഷി ലോഗുകൾ", 'logs_intro': "നിങ്ങളുടെ ദൈനംദിന പ്രവർത്തനങ്ങൾ, നിരീക്ഷണങ്ങൾ, കുറിപ്പുകൾ എന്നിവ ഇവിടെ രേഖപ്പെടുത്തുക.", 'new_log_label': "പുതിയ ലോഗ് എൻട്രി", 'save_log_button': "ലോഗ് സംരക്ഷിക്കുക", 'log_saved_success': "ലോഗ് എൻട്രി വിജയകരമായി സംരക്ഷിച്ചു!", 'past_logs_header': "📖 പഴയ എൻട്രികൾ", 'no_logs_message': "നിങ്ങൾക്ക് ഇതുവരെ ലോഗ് എൻട്രികളൊന്നും ഇല്ല.", 'delete_log_button': "ഇല്ലാതാക്കുക", 'log_deleted_success': "ലോഗ് ഇല്ലാതാക്കി!", 'record_log_prompt': "ഒരു വോയിസ് നോട്ട് അറ്റാച്ചുചെയ്യുക:", 'voice_note_recorded': "🎤 വോയിസ് നോട്ട് റെക്കോർഡുചെയ്‌തു, നിങ്ങളുടെ ലോഗിനൊപ്പം സംരക്ഷിക്കാൻ തയ്യാറാണ്.", 'attached_voice_note': "അറ്റാച്ചുചെയ്ത വോയിസ് നോട്ട്:",
        'upload_media_label': "ഫോട്ടോകളോ വീഡിയോകളോ അറ്റാച്ചുചെയ്യുക:", 'attached_images': "അറ്റാച്ചുചെയ്ത ചിത്രങ്ങൾ:", 'attached_videos': "അറ്റാച്ചുചെയ്ത വീഡിയോകൾ:",
        'forum_header': "👥 കമ്മ്യൂണിറ്റി ഫോറം", 'forum_intro': "മറ്റ് കർഷകരുമായി ബന്ധപ്പെടുക. ചോദ്യങ്ങൾ ചോദിക്കുക, പരിഹാരങ്ങൾ പങ്കിടുക, ഒരുമിച്ച് വളരുക.",
        'create_post_header': "ഒരു പുതിയ പോസ്റ്റ് സൃഷ്ടിക്കുക", 'post_title_label': "തലക്കെട്ട്", 'post_title_placeholder': "വ്യക്തവും സംക്ഷിപ്തവുമായ ഒരു തലക്കെട്ട് നൽകുക", 'post_content_label': "നിങ്ങളുടെ ചോദ്യം അല്ലെങ്കിൽ നിർദ്ദേശം", 'post_content_placeholder': "നിങ്ങളുടെ പ്രശ്നം വിവരിക്കുക അല്ലെങ്കിൽ നിങ്ങളുടെ അറിവ് വിശദമായി പങ്കിടുക...",
        'upload_image_label': "ഒരു ചിത്രം അപ്‌ലോഡ് ചെയ്യുക (ഓപ്ഷണൽ)", 'submit_post_button': "പോസ്റ്റ് സമർപ്പിക്കുക", 'post_success_message': "✅ നിങ്ങളുടെ പോസ്റ്റ് പ്രസിദ്ധീകരിച്ചു!",
        'all_posts_header': "എല്ലാ പോസ്റ്റുകളും", 'posted_by': "{username} പോസ്റ്റ് ചെയ്തത്, തീയതി: {date}",
        'like_button': "👍 ലൈക്ക് ചെയ്യുക ({count})", 'reply_button': "💬 മറുപടി നൽകുക", 'replies_header': "മറുപടികൾ", 'add_reply_placeholder': "ഒരു മറുപടി എഴുതുക...", 'submit_reply_button': "മറുപടി പോസ്റ്റ് ചെയ്യുക",
        'no_posts_message': "ഇതുവരെ പോസ്റ്റുകളൊന്നുമില്ല. ഒരു സംഭാഷണം ആരംഭിക്കുന്ന ആദ്യത്തെയാളാകൂ!", 'view_replies': "{count} മറുപടികൾ കാണുക", 'hide_replies': "മറുപടികൾ മറയ്ക്കുക",
        'soil_characteristics_heading': "🔬 സവിശേഷതകൾ", 'soil_suitable_crops_heading': "🌱 അനുയോജ്യമായ വിളകൾ", 'soil_model_not_found_warning': "⚠ മണ്ണ് പ്രവചന മോഡൽ കണ്ടെത്തിയില്ല. ദയവായി അഡ്മിനിസ്ട്രേറ്ററുമായി ബന്ധപ്പെടുക.",
        'stage_seedling': 'തൈകൾ', 'stage_vegetative': 'സസ്യജന്യം', 'stage_flowering': 'പൂവിടൽ', 'stage_fruiting': 'പഴവർഗ്ഗം', 'stage_maturity': 'പക്വത',
        'column_days_range': 'ദിവസങ്ങളുടെ പരിധി', 'column_fertilizer': 'ഉർവരകം', 'column_dosage': 'ഡോസേജ് (കി.ഗ്രാം/എക്കർ)', 'column_frequency': 'ആവൃത്തി', 'column_irrigation_depth': 'ജലസേചന ആഴം / അളവ്', 'column_irrigation_frequency': 'ജലസേചന ആവൃത്തി',
    },
    'ta': {
        'app_title': "கிருஷிசகி", 'go_to': "செல்க", 'go_button': "செல்", 'back_to_home': "முகப்பிற்கு திரும்பு", 'language_selector_label': "மொழி",
        'page_home': "முகப்பு", 'page_crop_recommendation': "பயிர் பரிந்துரை", 'page_yield_prediction': "விளைச்சல் கணிப்பு", 'page_crop_calendar': "பயிர் காலண்டர்", 'page_weather_report': "வானிலை அறிக்கை", 'page_leaf_disease_prediction': "இலை நோய் கணிப்பு", 'page_translator': "மொழிபெயர்ப்பாளர்", 'page_profile': "சுயவிவரம்", 'page_logs': "பதிவுகள்", 'page_community_forum': "சமூக மன்றம்",
        'quick_actions_title': "🚀 வினை நடவடிக்கைகள்", 'quick_action_crop_desc': "உங்கள் மண் மற்றும் வானிலை அடிப்படையில் ஏற்ற பயிர் பரிந்துரைகளைப் பெறுங்கள்.", 'quick_action_weather_desc': "உங்கள் பகுதியில் வானிலை சரிபார்த்து உங்கள் விவசாய நடவடிக்கைகளை திட்டமிடுங்கள்.",
        'page_crop_guide': "பயிர் வழிகாட்டி", 'page_pest_prediction': "பூச்சி கணிப்பு", 'page_soil_type_prediction': "மண் வகை கணிப்பு",
        'crop_guide_title': "பயிர் உரமிடுதல் மற்றும் நீர்ப்பாசன வழிகாட்டி", 'select_stage_label': "வளர்ச்சி நிலையைத் தேர்ந்தெடுக்கவும்", 'guide_table_header': "வழிகாட்டுதல் விவரங்கள்",
        'file_not_found_error': "கோப்பு காணப்படவில்லை. அது சரியான கோப்புறையில் உள்ளதா என்பதை உறுதிப்படுத்தவும்.",
        'home_title': "கிருஷிசகிக்கு வரவேற்கிறோம்! 🌱", 'home_subtitle': "நவீன விவசாயத்திற்கான உங்கள் AI-இயங்கும் உதவியாளர்.",
        'home_intro': "*கிருஷிசகி* விவசாயிகளுக்கு தரவு சார்ந்த நுண்ணறிவுகளை வழங்க வடிவமைக்கப்பட்டுள்ளது.\nபல்வேறு அம்சங்களை அணுக பக்க பட்டியில் உள்ள பிரிவுகளுக்கு செல்லவும்:\n- *🌾 பயிர் பரிந்துரை:* மண் மற்றும் சுற்றுச்சூழல் காரணிகளின் அடிப்படையில் பயிரிட சிறந்த பயிர்களுக்கான பரிந்துரைகளைப் பெறுங்கள்.\n- *📈 விளைச்சல் கணிப்பு:* வரலாற்றுத் தரவு மற்றும் விவசாய அளவுருக்களின் அடிப்படையில் உங்கள் பயிர் விளைச்சலைக் கணிக்கவும்.\n- *🗓 பயிர் காலண்டர்:* கேரளாவில் பல்வேறு பயிர்களுக்கான சிறந்த விதைப்பு மற்றும் அறுவடை நேரங்களைக் காண்க.\n- *🌦 வானிலை அறிக்கை:* உங்கள் இருப்பிடத்திற்கான நிகழ்நேர வானிலை அறிவிப்புகள் மற்றும் முன்னறிவிப்புகளைப் பெறுங்கள்.\n- *🌿 இலை நோய் கணிப்பு:* நோய்களை முன்கூட்டியே கண்டறிய ஒரு இலை படத்தை பதிவேற்றவும்.\n- *🌐 மொழிபெயர்ப்பாளர்:* விவசாய சொற்களையும் வாக்கியங்களையும் பல்வேறு இந்திய மொழிகளுக்கு இடையில் மொழிபெயர்க்கவும்.\n- *👥 சமூக மன்றம்:* மற்ற விவசாயிகளுடன் இணையுங்கள், கேள்விகளைக் கேளுங்கள், தீர்வுகளைப் பகிர்ந்து கொள்ளுங்கள்.",
        'crop_rec_header': "🌾 பயிர் பரிந்துரை", 'crop_rec_intro': "பயிர் பரிந்துரையைப் பெற பின்வரும் விவரங்களை உள்ளிடவும்.",
        'nitrogen_label': "நைட்ரஜன் (N)", 'phosphorus_label': "பாஸ்பரஸ் (P)", 'potassium_label': "பொட்டாசியம் (K)", 'temperature_label': "வெப்பநிலை (°C)", 'humidity_label': "ஈரப்பதம் (%)", 'ph_label': "pH மதிப்பு", 'rainfall_label': "மழையளவு (mm)",
        'recommend_crop_button': "பயிரைப் பரிந்துரைக்கவும்", 'recommendation_success': "🎉 பரிந்துரைக்கப்பட்ட பயிர் *{crop}* ஆகும்.",
        'crop_info_header': "{crop} பற்றி மேலும் தகவல்", 'description_heading': "📖 விளக்கம்", 'irrigation_heading': "💧 பாசன குறிப்புகள்", 'conditions_heading': "☀️ சிறந்த நிபந்தனைகள்", 'pests_heading': "🐛 பொதுவான பூச்சிகள் மற்றும் நோய்கள்",
        'yield_pred_header': "📈 விளைச்சல் கணிப்பு", 'yield_pred_intro': "கேரளாவின் ஒரு குறிப்பிட்ட பகுதிக்கான பயிர் விளைச்சலைக் கணிக்க கீழே உள்ள விவரங்களை நிரப்பவும்.",
        'year_label': "ஆண்டு", 'crop_label': "பயிர்", 'season_label': "பருவம்", 'area_label': "பரப்பளவு (ஹெக்டேரில்)",
        'predict_yield_button': "விளைச்சலைக் கணிக்கவும்", 'yield_prediction_success': "🌾 கணிக்கப்பட்ட பயிர் விளைச்சல் தோராயமாக *{yield_val:.2f} கிகி/ஹெக்டேர்* ஆகும்.",
        'market_price_header': "💰 மதிப்பிடப்பட்ட சந்தை மதிப்பு", 'fetching_prices_spinner': "சமீபத்திய சந்தை விலைகளைப் பெறுகிறது...", 'market_price_label': "சராசரி சந்தை விலை ( குவிண்டாலுக்கு)", 'total_value_label': "மொத்த மதிப்பிடப்பட்ட மதிப்பு", 'profit_header': "💵 மதிப்பிடப்பட்ட லாபம்", 'cost_per_ha_label': "விவசாய செலவு ஒரு ஹெக்டேருக்கு (₹)", 'total_cost_label': "மொத்த விவசாய செலவு", 'total_profit_label': "மொத்த மதிப்பிடப்பட்ட லாபம்", 'price_not_available': "{crop} க்கான விலை தரவு தற்போது கிடைக்கவில்லை.", 'price_api_error': "தற்போது சந்தை விலை தரவைப் பெற முடியவில்லை.",
        'crop_calendar_header': "🗓 பயிர் விதைப்பு மற்றும் அறுவடை காலண்டர்", 'select_crop_label': "ஒரு பயிரைத் தேர்ந்தெடுக்கவும்",
        'schedule_for_crop': "📅 {crop}-க்கான அட்டவணை", 'sowing_start_label': "விதைப்பு தொடக்கம்", 'sowing_end_label': "விதைப்பு முடிவு", 'harvest_start_label': "அறுவடை தொடக்கம்", 'harvest_end_label': "அறுவடை முடிவு", 'fertiliser_label': "உரம் (N:P:K கிலோ/ஹெக்டேர்)", 'duration_label': "காலம்",
        'weather_report_header': "🌦 வானிலை அறிக்கை", 'enter_city_label': "உங்கள் நகரத்தின் பெயரை உள்ளிடவும்", 'get_weather_button': "வானிலை பெறுக",
        'current_weather_in': "{location}-ல் தற்போதைய வானிலை", 'temperature_metric': "வெப்பநிலை", 'feels_like_metric': "உணர்வது போல்", 'humidity_metric': "ஈரப்பதம்", 'description_label': "விளக்கம்:", 'wind_label': "காற்று:", 'precipitation_label': "மழைப்பொழிவு:",
        'forecast_header': "🗓 3-நாள் முன்னறிவிப்பு", 'avg_temp_label': "சராசரி வெப்பநிலை:", 'max_temp_label': "அதிகபட்ச வெப்பநிலை:", 'min_temp_label': "குறைந்தபட்ச வெப்பநிலை:", 'sunrise_label': "சூரிய உதயம்:", 'sunset_label': "சூரிய அஸ்தமனம்:", 'outlook_label': "கண்ணோட்டம்:",
        'error_parsing_weather': "வானிலை தரவை பாகுபடுத்த முடியவில்லை. தயவுசெய்து நகரத்தின் பெயரைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும். பிழை: {e}", 'enter_city_warning': "⚠ தயவுசெய்து ஒரு நகரத்தின் பெயரை உள்ளிடவும்.",
        'translator_header': "🌐 விவசாயி மொழி மொழிபெயர்ப்பாளர்", 'translator_intro': "✍ மொழிபெயர்க்க உரையை உள்ளிடவும்", 'translate_button': "🔄 மொழிபெயர்க்கவும்",
        'translation_failed_error': "❌ மொழிபெயர்ப்பு தோல்வியடைந்தது: {e}", 'enter_text_warning': "⚠ தயவுசெய்து சில உரையை உள்ளிடவும்.",
        'ai_chatbot_header': "🤖 AI அரட்டைப்பெட்டி", 'ai_chatbot_intro': "இந்த அம்சம் விரைவில் வருகிறது! உங்கள் விவசாய கேள்விகளுக்கு உதவ ஒரு சக்திவாய்ந்த AI அரட்டைப்பெட்டிக்காக காத்திருங்கள்.",
        'languages_supported_label': "ஆதரிக்கப்படும் மொழிகள்", 'farmers_helped_label': "உதவப்பட்ட விவசாயிகள்", 'crops_covered_label': "உள்ளடக்கப்பட்ட பயிர்கள்", 'recommendations_given_label': "AI பரிந்துரைகள்",
        'leaf_disease_header': "🌿 இலை நோய் கணிப்பு", 'leaf_disease_intro': "நோய்களைச் சரிபார்க்க தாவர இலையின் படத்தை பதிவேற்றவும்.", 'model_not_found_warning': "⚠ நோய் கணிப்பு மாதிரி காணப்படவில்லை. தயவுசெய்து நிர்வாகியைத் தொடர்பு கொள்ளவும்.",
        'uploader_label': "இலையின் படத்தைத் தேர்ந்தெடுக்கவும்...", 'uploaded_image_caption': "பதிவேற்றிய படம்.", 'predict_disease_button': "🔍 நோயைக் கணிக்கவும்",
        'predicting_spinner': "இலையை பகுப்பாய்வு செய்கிறது...", 'prediction_result': "✅ கணிப்பு: *{disease}*",
        'remedy_info_header': "{disease}-க்கான சிகிச்சை மற்றும் மேலாண்மை", 'disease_description_heading': "ℹ️ நோய் பற்றி", 'organic_remedies_heading': "🌿 இயற்கை வைத்தியம்", 'chemical_remedies_heading': "🧪 இரசாயன வைத்தியம்", 'healthy_plant_message': "ஒரு நல்ல செய்தி! உங்கள் ஆலை ஆரோக்கியமாக தெரிகிறது. நல்ல விவசாய நடைமுறைகளைத் தொடரவும்.",
        'soil_prediction_header': "🏞️ மண் வகை கணிப்பு", 'soil_prediction_intro': "மண்ணின் வகையை அடையாளம் கண்டு அதன் பண்புகளைப் பற்றி அறிய மண்ணின் படத்தை பதிவேற்றவும்.", 'upload_soil_image_label': "மண்ணின் படத்தைத் தேர்ந்தெடுக்கவும்...",
        'predict_soil_button': "🔍 மண் வகையை கணிக்கவும்", 'analyzing_soil_spinner': "மண்ணை பகுப்பாய்வு செய்கிறது...", 'soil_prediction_result': "✅ கணிக்கப்பட்ட மண் வகை: *{soil_type}*", 'soil_info_header': "{soil_type} பற்றிய தகவல்",
        'pest_prediction_header': "படத்திலிருந்து பூச்சிகளை அடையாளம் காணுதல்", 'upload_image_pest': "பூச்சி அல்லது பாதிக்கப்பட்ட இலையின் படத்தை பதிவேற்றவும்", 'prediction_pest': "கணிக்கப்பட்ட பூச்சி", 'model_loading_error': "பூச்சி முன்கணிப்பு மாதிரியை ஏற்ற முடியவில்லை. இணைய இணைப்பைச் சரிபார்க்கவும்.",
        'remedy_info_header_pest': "{pest}-க்கான சிகிச்சை மற்றும் மேலாண்மை", 'pest_description_heading': "ℹ️ பூச்சி பற்றி", 'no_remedy_info': "{pest}-க்கான தீர்வு தகவல் இன்னும் கிடைக்கவில்லை.",
        'feedback_header': "பின்னூட்டம்", 'rating_label': "உங்கள் அனுபவத்தை மதிப்பிடுங்கள்:", 'submit_button': "சமர்ப்பிக்கவும்", 'feedback_success': "உங்கள் கருத்துக்கு நன்றி!",
        'contact_header': "தொடர்பு கொள்ளவும்",
        'weather_api_key_error': "API விசை பிழை: {message}. secrets.toml இல் உங்கள் OpenWeatherMap API விசையைச் சரிபார்க்கவும். புதிய விசைகள் செயல்பட சில மணிநேரங்கள் ஆகலாம்.", 'weather_city_not_found_error': "நகரம் காணப்படவில்லை பிழை: {message}. நகரத்தின் பெயரின் எழுத்துப்பிழையை சரிபார்க்கவும்.", 'weather_api_generic_error': "API பிழை ({cod}): {message}",
        'login_page_title': "கிருஷிசகிக்கு வரவேற்கிறோம் 🌱", 'login_tab': "உள்நுழைக", 'signup_tab': "பதிவு செய்க",
        'username_label': "பயனர்பெயர்", 'password_label': "கடவுச்சொல்", 'login_button': "உள்நுழைக", 'signup_button': "பதிவு செய்க", 'logout_button': "வெளியேறு",
        'login_error': "தவறான பயனர்பெயர் அல்லது கடவுச்சொல்.", 'signup_success': "பதிவு வெற்றிகரமாக முடிந்தது! தயவுசெய்து உள்நுழையவும்.", 'signup_error_exists': "பயனர்பெயர் ஏற்கனவே உள்ளது. தயவுசெய்து மற்றொன்றைத் தேர்ந்தெடுக்கவும்.", 'signup_error_general': "பதிவின் போது ஒரு பிழை ஏற்பட்டது.",
        'welcome_back': "மீண்டும் வருக, {username}!",
        'profile_header': "👤 எனது சுயவிவரம்", 'profile_subheader': "உங்கள் கணக்கு விவரங்கள் மற்றும் அமைப்புகளை நிர்வகிக்கவும்.",
        'full_name_label': "முழு பெயர்", 'contact_number_label': "தொடர்பு எண்", 'new_username_label': "புதிய பயனர்பெயர்", 'new_password_label': "புதிய கடவுச்சொல்", 'confirm_password_label': "புதிய கடவுச்சொல்லை உறுதிப்படுத்தவும்", 'profile_pic_label': "சுயவிவரப் படம்",
        'update_profile_button': "சுயவிவரத்தைப் புதுப்பிக்கவும்", 'profile_update_success': "✅ சுயவிவரம் வெற்றிகரமாக புதுப்பிக்கப்பட்டது!", 'profile_update_error': "❌ சுயவிவரத்தைப் புதுப்பிக்க முடியவில்லை. பயனர்பெயர் ஏற்கனவே இருக்கலாம்.", 'password_mismatch_error': "கடவுச்சொற்கள் பொருந்தவில்லை.",
        'danger_zone_header': "🚨 ஆபத்து மண்டலம்", 'delete_account_button': "எனது கணக்கை நீக்கு", 'delete_account_warning': "இந்த நடவடிக்கை மாற்ற முடியாதது. உங்கள் எல்லா தரவும் நிரந்தரமாக நீக்கப்படும்.", 'delete_confirmation_label': "உறுதிப்படுத்த, கீழே உள்ள பெட்டியில் `DELETE` என தட்டச்சு செய்யவும்.",
        'delete_account_confirm_button': "ஆம், எனது கணக்கை நீக்க விரும்புகிறேன்", 'account_deleted_success': "உங்கள் கணக்கு வெற்றிகரமாக நீக்கப்பட்டது.", 'incorrect_delete_confirmation': "தவறான உறுதிப்படுத்தல் உரை.",
        'logs_header': "எனது விவசாய பதிவுகள்", 'logs_intro': "உங்கள் தினசரி நடவடிக்கைகள், அவதானிப்புகள் மற்றும் குறிப்புகளை இங்கே பதிவு செய்யவும்.", 'new_log_label': "புதிய பதிவு நுழைவு", 'save_log_button': "பதிவைச் சேமி", 'log_saved_success': "பதிவு நுழைவு வெற்றிகரமாக சேமிக்கப்பட்டது!", 'past_logs_header': "📖 கடந்த பதிவுகள்", 'no_logs_message': "உங்களிடம் இன்னும் பதிவு உள்ளீடுகள் எதுவும் இல்லை.", 'delete_log_button': "அழி", 'log_deleted_success': "பதிவு நீக்கப்பட்டது!", 'record_log_prompt': "ஒரு குரல் குறிப்பை இணைக்கவும்:", 'voice_note_recorded': "🎤 குரல் குறிப்பு பதிவு செய்யப்பட்டு உங்கள் பதிவில் சேமிக்க தயாராக உள்ளது.", 'attached_voice_note': "இணைக்கப்பட்ட குரல் குறிப்பு:",
        'upload_media_label': "புகைப்படங்கள் அல்லது வீடியோக்களை இணைக்கவும்:", 'attached_images': "இணைக்கப்பட்ட படங்கள்:", 'attached_videos': "இணைக்கப்பட்ட வீடியோக்கள்:",
        'forum_header': "👥 சமூக மன்றம்", 'forum_intro': "மற்ற விவசாயிகளுடன் இணையுங்கள். கேள்விகளைக் கேளுங்கள், தீர்வுகளைப் பகிர்ந்து கொள்ளுங்கள், ஒன்றாக வளருங்கள்.",
        'create_post_header': "புதிய இடுகையை உருவாக்கவும்", 'post_title_label': "தலைப்பு", 'post_title_placeholder': "தெளிவான மற்றும் சுருக்கமான தலைப்பை உள்ளிடவும்", 'post_content_label': "உங்கள் கேள்வி அல்லது குறிப்பு", 'post_content_placeholder': "உங்கள் சிக்கலை விவரிக்கவும் அல்லது உங்கள் அறிவை விரிவாகப் பகிரவும்...",
        'upload_image_label': "ஒரு படத்தைப் பதிவேற்றவும் (விருப்பத்தேர்வு)", 'submit_post_button': "இடுகையைச் சமர்ப்பிக்கவும்", 'post_success_message': "✅ உங்கள் இடுகை வெளியிடப்பட்டது!",
        'all_posts_header': "அனைத்து இடுகைகளும்", 'posted_by': "{username} ஆல் {date} அன்று வெளியிடப்பட்டது", 'like_button': "👍 விரும்புக ({count})", 'reply_button': "💬 பதிலளி", 'replies_header': "பதில்கள்", 'add_reply_placeholder': "பதிலை எழுதுங்கள்...",
        'submit_reply_button': "பதிலை இடுகையிடவும்", 'no_posts_message': "இன்னும் இடுகைகள் இல்லை. உரையாடலைத் தொடங்குபவர் நீங்களாக இருங்கள்!", 'view_replies': "{count} பதில்களைக் காண்க", 'hide_replies': "பதில்களை மறை",
        'soil_characteristics_heading': "🔬 பண்புகள்", 'soil_suitable_crops_heading': "🌱 பொருத்தமான பயிர்கள்", 'soil_model_not_found_warning': "⚠ மண் கணிப்பு மாதிரி காணப்படவில்லை. தயவுசெய்து நிர்வாகியைத் தொடர்பு கொள்ளவும்.",
        'stage_seedling': 'முளைப்பு', 'stage_vegetative': 'தாவர', 'stage_flowering': 'பூக்கும்', 'stage_fruiting': 'கனி', 'stage_maturity': 'முதிர்ச்சி',
        'column_days_range': 'நாட்கள் வரம்பு', 'column_fertilizer': 'உரம்', 'column_dosage': 'அளவு (கி.கி/எக்டர்)', 'column_frequency': 'அதிர்வெண்', 'column_irrigation_depth': 'நீர்ப்பாசன ஆழம் / அளவு', 'column_irrigation_frequency': 'நீர்ப்பாசன அதிர்வெண்',
    },
    'te': {
        'app_title': "కృషిసఖి", 'go_to': "వెళ్ళండి", 'go_button': "వెళ్ళండి", 'back_to_home': "హోమ్‌కి తిరిగి", 'language_selector_label': "భాష",
        'page_home': "హోమ్", 'page_crop_recommendation': "పంట సిఫార్సు", 'page_yield_prediction': "దిగుబడి అంచనా", 'page_crop_calendar': "పంట క్యాలెండర్", 'page_weather_report': "వాతావరణ నివేదిక", 'page_leaf_disease_prediction': "ఆకు వ్యాధి πρόβλεψη", 'page_translator': "అనువాదకుడు", 'page_profile': "ప్రొఫైల్", 'page_logs': "లాగ్‌లు", 'page_community_forum': "కమ్యూనిటీ ఫోరమ్",
        'quick_actions_title': "🚀 తక్షణ చర్యలు", 'quick_action_crop_desc': "మీ మట్టి మరియు వాతావరణం ఆధారంగా సరిపోయే పంట సిఫార్సులను పొందండి.", 'quick_action_weather_desc': "మీ ప్రాంతంలోని వాతావరణాన్ని తనిఖీ చేసి మీ వ్యవసాయ కార్యాచరణలను ప్రణాళిక చేయండి.",
        'page_crop_guide': "పంట గైడ్", 'page_pest_prediction': "కీటక అంచనా", 'page_soil_type_prediction': "నేల రకం అంచనా",
        'crop_guide_title': "పంట ఫలదీకరణ మరియు నీటిపారుదల గైడ్", 'select_stage_label': "పెరుగుదల దశను ఎంచుకోండి", 'guide_table_header': "మార్గదర్శక వివరాలు",
        'file_not_found_error': "ఫైల్ కనుగొనబడలేదు. దయచేసి అది సరైన ఫోల్డర్‌లో ఉందని నిర్ధారించుకోండి.",
        'home_title': "కృషిసఖికి స్వాగతం! 🌱", 'home_subtitle': "ఆధునిక వ్యవసాయం కోసం మీ AI-ఆధారిత సహాయకుడు.",
        'home_intro': "*కృషిసఖి* రైతులకు డేటా-ఆధారిత అంతర్దృష్టులను అందించడానికి రూపొందించబడింది.\nవివిధ ఫీచర్లను యాక్సెస్ చేయడానికి సైడ్‌బార్‌లోని విభాగాల ద్వారా నావిగేట్ చేయండి:\n- *🌾 పంట సిఫార్సు:* నేల మరియు పర్యావరణ కారకాల ఆధారంగా నాటడానికి ఉత్తమమైన పంటల కోసం సూచనలను పొందండి.\n- *📈 దిగుబడి అంచనా:* చారిత్రక డేటా మరియు వ్యవసాయ పారామితుల ఆధారంగా మీ పంట దిగుబడిని అంచనా వేయండి.\n- *🗓 పంట క్యాలెండర్:* కేరళలోని వివిధ పంటలకు అనువైన విత్తనాలు మరియు కోత సమయాలను వీక్షించండి.\n- *🌦 వాతావరణ నివేదిక:* మీ స్థానం కోసం నిజ-సమయ వాతావరణ నవీకరణలు మరియు అంచనాలను పొందండి.\n- *🌿 ఆకు వ్యాధి πρόβλεψη:* వ్యాధులను ముందుగా గుర్తించడానికి ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి.\n- *🌐 అనువాదకుడు:* వ్యవసాయ పదాలు మరియు వాక్యాలను వివిధ భారతీయ భాషల మధ్య అనువదించండి.\n- *👥 కమ్యూనిటీ ఫోరమ్:* ఇతర రైతులతో కనెక్ట్ అవ్వండి, ప్రశ్నలు అడగండి మరియు పరిష్కారాలను పంచుకోండి.",
        'crop_rec_header': "🌾 పంట సిఫార్సు", 'crop_rec_intro': "పంట సిఫార్సు పొందడానికి క్రింది వివరాలను నమోదు చేయండి.",
        'nitrogen_label': "నత్రజని (N)", 'phosphorus_label': "భాస్వరం (P)", 'potassium_label': "పొటాషియం (K)", 'temperature_label': "ఉష్ణోగ్రత (°C)", 'humidity_label': "తేమ (%)", 'ph_label': "pH విలువ", 'rainfall_label': "వర్షపాతం (mm)",
        'recommend_crop_button': "పంటను సిఫార్సు చేయండి", 'recommendation_success': "🎉 సిఫార్సు చేయబడిన పంట *{crop}*.",
        'crop_info_header': "{crop} గురించి మరింత సమాచారం", 'description_heading': "📖 వివరణ", 'irrigation_heading': "💧 నీటిపారుదల చిట్కాలు", 'conditions_heading': "☀️ ఆదర్శ పరిస్థితులు", 'pests_heading': "🐛 సాధారణ తెగుళ్లు & వ్యాధులు",
        'yield_pred_header': "📈 దిగుబడి అంచనా", 'yield_pred_intro': "కేరళలోని ఒక నిర్దిష్ట ప్రాంతానికి పంట దిగుబడిని అంచనా వేయడానికి క్రింది వివరాలను పూరించండి.",
        'year_label': "సంవత్సరం", 'crop_label': "పంట", 'season_label': "సీజన్", 'area_label': "విస్తీర్ణం (హెక్టార్లలో)",
        'predict_yield_button': "దిగుబడిని అంచనా వేయండి", 'yield_prediction_success': "🌾 అంచనా వేయబడిన పంట దిగుబడి సుమారుగా *{yield_val:.2f} కేజీలు/హెక్టారు*.",
        'market_price_header': "💰 అంచనా వేయబడిన మార్కెట్ విలువ", 'fetching_prices_spinner': "తాజా మార్కెట్ ధరలను పొందుతోంది...", 'market_price_label': "సగటు మార్కెట్ ధర (క్వింటాల్‌కు)", 'total_value_label': "మొత్తం అంచనా విలువ", 'profit_header': "💵 అంచనా వేయబడిన లాభం", 'cost_per_ha_label': "వ్యవసాయ ఖర్చు హెక్టారుకు (₹)", 'total_cost_label': "మొత్తం వ్యవసాయ ఖర్చు", 'total_profit_label': "మొత్తం అంచనా లాభం", 'price_not_available': "{crop} కోసం ధర డేటా ప్రస్తుతం అందుబాటులో లేదు.", 'price_api_error': "ప్రస్తుతం మార్కెట్ ధర డేటాను పొందడం సాధ్యపడలేదు.",
        'crop_calendar_header': "🗓 పంట విత్తనాలు మరియు కోత క్యాలెండర్", 'select_crop_label': "ఒక పంటను ఎంచుకోండి",
        'schedule_for_crop': "📅 {crop} కోసం షెడ్యూల్", 'sowing_start_label': "విత్తనాలు ప్రారంభం", 'sowing_end_label': "విత్తనాలు ముగింపు", 'harvest_start_label': "కోత ప్రారంభం", 'harvest_end_label': "కోత ముగింపు", 'fertiliser_label': "ఎరువులు (N:P:K కేజీలు/హెక్టారు)", 'duration_label': "వ్యవధి",
        'weather_report_header': "🌦 వాతావరణ నివేదిక", 'enter_city_label': "మీ నగరం పేరును నమోదు చేయండి", 'get_weather_button': "వాతావరణాన్ని పొందండి",
        'current_weather_in': "{location}లో ప్రస్తుత వాతావరణం", 'temperature_metric': "ఉష్ణోగ్రత", 'feels_like_metric': "అనిపిస్తుంది", 'humidity_metric': "తేమ", 'description_label': "వివరణ:", 'wind_label': "గాలి:", 'precipitation_label': "వర్షపాతం:",
        'forecast_header': "🗓 3-రోజుల అంచనా", 'avg_temp_label': "సగటు ఉష్ణోగ్రత:", 'max_temp_label': "గరిష్ట ఉష్ణోగ్రత:", 'min_temp_label': "కనిష్ట ఉష్ణోగ్రత:", 'sunrise_label': "సూర్యోదయం:", 'sunset_label': "సూర్యాస్తమయం:", 'outlook_label': "అవలోకనం:",
        'error_parsing_weather': "వాతావరణ డేటాను పార్స్ చేయడంలో విఫలమైంది. దయచేసి నగరం పేరును తనిఖీ చేసి మళ్లీ ప్రయత్నించండి. దోషం: {e}", 'enter_city_warning': "⚠ దయచేసి ఒక నగరం పేరును నమోదు చేయండి.",
        'translator_header': "🌐 రైతు భాషా అనువాదకుడు", 'translator_intro': "✍ అనువదించడానికి వచనాన్ని నమోదు చేయండి", 'translate_button': "🔄 అనువదించండి",
        'translation_failed_error': "❌ అనువాదం విఫలమైంది: {e}", 'enter_text_warning': "⚠ దయచేసి కొంత వచనాన్ని నమోదు చేయండి.",
        'ai_chatbot_header': "🤖 AI చాట్‌బాట్", 'ai_chatbot_intro': "ఈ ఫీచర్ త్వరలో వస్తుంది! మీ వ్యవసాయ ప్రశ్నలకు సహాయపడటానికి ఒక శక్తివంతమైన AI చాట్‌బాట్ కోసం వేచి ఉండండి.",
        'languages_supported_label': "మద్దతు ఉన్న భాషలు", 'farmers_helped_label': "సహాయపడిన రైతులు", 'crops_covered_label': "కవర్ చేయబడిన పంటలు", 'recommendations_given_label': "AI సిఫార్సులు",
        'leaf_disease_header': "🌿 ఆకు వ్యాధి πρόβλεψη", 'leaf_disease_intro': "వ్యాధులను తనిఖీ చేయడానికి మొక్క ఆకు యొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి.", 'model_not_found_warning': "⚠ వ్యాధి πρόβλεψη నమూనా కనుగొనబడలేదు. దయచేసి నిర్వాహకుడిని సంప్రదించండి.",
        'uploader_label': "ఆకు చిత్రాన్ని ఎంచుకోండి...", 'uploaded_image_caption': "అప్‌లోడ్ చేయబడిన చిత్రం.", 'predict_disease_button': "🔍 వ్యాధిని అంచనా వేయండి",
        'predicting_spinner': "ఆకును విశ్లేషిస్తోంది...", 'prediction_result': "✅ అంచనా: *{disease}*",
        'remedy_info_header': "{disease} కోసం చికిత్స మరియు నిర్వహణ", 'disease_description_heading': "ℹ️ వ్యాధి గురించి", 'organic_remedies_heading': "🌿 సేంద్రీయ నివారణలు", 'chemical_remedies_heading': "🧪 రసాయన నివారణలు", 'healthy_plant_message': "శుభవార్త! మీ మొక్క ఆరోగ్యంగా కనిపిస్తుంది. మంచి వ్యవసాయ పద్ధతులను కొనసాగించండి.",
        'soil_prediction_header': "🏞️ నేల రకం అంచనా", 'soil_prediction_intro': "నేల రకాన్ని గుర్తించడానికి మరియు దాని లక్షణాల గురించి తెలుసుకోవడానికి నేల చిత్రాన్ని అప్‌లోడ్ చేయండి.", 'upload_soil_image_label': "నేల చిత్రాన్ని ఎంచుకోండి...",
        'predict_soil_button': "🔍 నేల రకాన్ని అంచనా వేయండి", 'analyzing_soil_spinner': "నేలను విశ్లేషిస్తోంది...", 'soil_prediction_result': "✅ అంచనా వేయబడిన నేల రకం: *{soil_type}*", 'soil_info_header': "{soil_type} గురించి సమాచారం",
        'pest_prediction_header': "చిత్రం నుండి కీటకాలను గుర్తించడం", 'upload_image_pest': "కీటకం లేదా ప్రభావిత ఆకు యొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి", 'prediction_pest': "అంచనా వేసిన కీటకం", 'model_loading_error': "కీటక అంచనా మోడల్‌ను లోడ్ చేయలేకపోయాము. దయచేసి ఇంటర్నెట్ కనెక్షన్‌ను తనిఖీ చేయండి.",
        'remedy_info_header_pest': "{pest} కోసం చికిత్స మరియు నిర్వహణ", 'pest_description_heading': "ℹ️ కీటకం గురించి", 'no_remedy_info': "{pest} కోసం నివారణ సమాచారం ఇంకా అందుబాటులో లేదు.",
        'feedback_header': "అభిప్రాయం", 'rating_label': "మీ అనుభవాన్ని రేట్ చేయండి:", 'submit_button': "సమర్పించు", 'feedback_success': "మీ అభిప్రాయానికి ధన్యవాదాలు!",
        'contact_header': "సంప్రదించండి",
        'weather_api_key_error': "API కీ లోపం: {message}. దయచేసి secrets.tomlలో మీ OpenWeatherMap API కీని తనిఖీ చేయండి. కొత్త కీలు సక్రియం కావడానికి కొన్ని గంటల సమయం పట్టవచ్చు.", 'weather_city_not_found_error': "నగరం కనుగొనబడలేదు లోపం: {message}. దయచేసి నగరం పేరు యొక్క స్పెల్లింగ్‌ను తనిఖీ చేయండి.", 'weather_api_generic_error': "API లోపం ({cod}): {message}",
        'login_page_title': "కృషిసఖికి స్వాగతం 🌱", 'login_tab': "లాగిన్", 'signup_tab': "నమోదు చేసుకోండి",
        'username_label': "వినియోగదారు పేరు", 'password_label': "పాస్‌వర్డ్", 'login_button': "లాగిన్", 'signup_button': "నమోదు చేసుకోండి", 'logout_button': "లాగ్ అవుట్",
        'login_error': "తప్పు వినియోగదారు పేరు లేదా పాస్‌వర్డ్.", 'signup_success': "నమోదు విజయవంతమైంది! దయచేసి లాగిన్ చేయండి.", 'signup_error_exists': "వినియోగదారు పేరు ఇప్పటికే ఉంది. దయచేసి మరొకదాన్ని ఎంచుకోండి.", 'signup_error_general': "నమోదు సమయంలో ఒక లోపం సంభవించింది.",
        'welcome_back': "తిరిగి స్వాగతం, {username}!",
        'profile_header': "👤 నా ప్రొఫైల్", 'profile_subheader': "మీ ఖాతా వివరాలు మరియు సెట్టింగ్‌లను నిర్వహించండి.",
        'full_name_label': "పూర్తి పేరు", 'contact_number_label': "సంప్రదింపు సంఖ్య", 'new_username_label': "కొత్త వినియోగదారు పేరు", 'new_password_label': "కొత్త పాస్వర్డ్", 'confirm_password_label': "కొత్త పాస్‌వర్డ్‌ను నిర్ధారించండి", 'profile_pic_label': "ప్రొఫైల్ చిత్రం",
        'update_profile_button': "ప్రొఫైల్‌ను నవీకరించండి", 'profile_update_success': "✅ ప్రొఫైల్ విజయవంతంగా నవీకరించబడింది!", 'profile_update_error': "❌ ప్రొఫైల్‌ను నవీకరించడంలో విఫలమైంది. వినియోగదారు పేరు ఇప్పటికే ఉండవచ్చు.", 'password_mismatch_error': "పాస్వర్డ్లు సరిపోలడం లేదు.",
        'danger_zone_header': "🚨 ప్రమాదకర ప్రాంతం", 'delete_account_button': "నా ఖాతాను తొలగించు", 'delete_account_warning': "ఈ చర్య మార్చలేనిది. మీ డేటా అంతా శాశ్వతంగా తొలగించబడుతుంది.", 'delete_confirmation_label': "ధృవీకరించడానికి, దయచేసి దిగువ పెట్టెలో `DELETE` అని టైప్ చేయండి.",
        'delete_account_confirm_button': "అవును, నేను నా ఖాతాను తొలగించాలనుకుంటున్నాను", 'account_deleted_success': "మీ ఖాతా విజయవంతంగా తొలగించబడింది.", 'incorrect_delete_confirmation': "తప్పు నిర్ధారణ వచనం.",
        'logs_header': "నా వ్యవసాయ లాగ్‌లు", 'logs_intro': "మీ రోజువారీ కార్యకలాపాలు, పరిశీలనలు మరియు గమనికలను ఇక్కడ రికార్డ్ చేయండి.", 'new_log_label': "కొత్త లాగ్ ఎంట్రీ", 'save_log_button': "లాగ్‌ను సేవ్ చేయండి", 'log_saved_success': "లాగ్ ఎంట్రీ విజయవంతంగా సేవ్ చేయబడింది!", 'past_logs_header': "📖 గత ఎంట్రీలు", 'no_logs_message': "మీకు ఇంకా లాగ్ ఎంట్రీలు లేవు.", 'delete_log_button': "తొలగించు", 'log_deleted_success': "లాగ్ తొలగించబడింది!", 'record_log_prompt': "ఒక వాయిస్ నోట్‌ను అటాచ్ చేయండి:", 'voice_note_recorded': "🎤 వాయిస్ నోట్ రికార్డ్ చేయబడింది మరియు మీ లాగ్‌తో సేవ్ చేయడానికి సిద్ధంగా ఉంది.", 'attached_voice_note': "అటాచ్ చేసిన వాయిస్ నోట్:",
        'upload_media_label': "ఫోటోలు లేదా వీడియోలను అటాచ్ చేయండి:", 'attached_images': "అటాచ్ చేసిన చిత్రాలు:", 'attached_videos': "అటాచ్ చేసిన వీడియోలు:",
        'forum_header': "👥 కమ్యూనిటీ ఫోరమ్", 'forum_intro': "ఇతర రైతులతో కనెక్ట్ అవ్వండి. ప్రశ్నలు అడగండి, పరిష్కారాలు పంచుకోండి మరియు కలిసి పెరగండి.",
        'create_post_header': "కొత్త పోస్ట్‌ను సృష్టించండి", 'post_title_label': "శీర్షిక", 'post_title_placeholder': "స్పష్టమైన మరియు సంక్షిప్త శీర్షికను నమోదు చేయండి", 'post_content_label': "మీ ప్రశ్న లేదా చిట్కా", 'post_content_placeholder': "మీ సమస్యను వివరించండి లేదా మీ జ్ఞానాన్ని వివరంగా పంచుకోండి...",
        'upload_image_label': "ఒక చిత్రాన్ని అప్‌లోడ్ చేయండి (ఐచ్ఛికం)", 'submit_post_button': "పోస్ట్‌ను సమర్పించండి", 'post_success_message': "✅ మీ పోస్ట్ ప్రచురించబడింది!",
        'all_posts_header': "అన్ని పోస్ట్‌లు", 'posted_by': "{username} ద్వారా {date}న పోస్ట్ చేయబడింది", 'like_button': "👍 లైక్ ({count})", 'reply_button': "💬 ప్రత్యుత్తరం", 'replies_header': "ప్రత్యుత్తరాలు", 'add_reply_placeholder': "ఒక ప్రత్యుత్తరం వ్రాయండి...",
        'submit_reply_button': "ప్రత్యుత్తరాన్ని పోస్ట్ చేయండి", 'no_posts_message': "ఇంకా పోస్ట్‌లు లేవు. సంభాషణను ప్రారంభించిన మొదటి వ్యక్తిగా ఉండండి!", 'view_replies': "{count} ప్రత్యుత్తరాలను వీక్షించండి", 'hide_replies': "ప్రత్యుత్తరాలను దాచండి",
        'soil_characteristics_heading': "🔬 లక్షణాలు", 'soil_suitable_crops_heading': "🌱 తగిన పంటలు", 'soil_model_not_found_warning': "⚠ నేల అంచనా నమూనా కనుగొనబడలేదు. దయచేసి నిర్వాహకుడిని సంప్రదించండి.",
        'stage_seedling': 'మొలక', 'stage_vegetative': 'సస్యజన్యం', 'stage_flowering': 'పూవులు', 'stage_fruiting': 'పండ్లు', 'stage_maturity': 'పరిపక్వత',
        'column_days_range': 'రోజుల పరిధి', 'column_fertilizer': 'ఎరువు', 'column_dosage': 'మోతాదు (కి.గ్రా/ఎకరా)', 'column_frequency': 'తరచుదనం', 'column_irrigation_depth': 'నీటిపారుదల లోతు / పరిమాణం', 'column_irrigation_frequency': 'నీటిపారుదల తరచుదనం',
    },
    'bn': {
        'app_title': "কৃষিসখী", 'go_to': "যান", 'go_button': "যান", 'back_to_home': "হোমে ফিরে যান", 'language_selector_label': "ভাষা",
        'page_home': "হোম", 'page_crop_recommendation': "ফসল সুপারিশ", 'page_yield_prediction': "উৎপাদন ভবিষ্যদ্বাণী", 'page_crop_calendar': "ফসল ক্যালেন্ডার", 'page_weather_report': "আবহাওয়া রিপোর্ট", 'page_leaf_disease_prediction': "পাতা রোগ ভবিষ্যদ্বাণী", 'page_translator': "অনুবাদক", 'page_profile': "প্রোফাইল", 'page_logs': "লগ", 'page_community_forum': "কমিউনিটি ফোরাম",
        'quick_actions_title': "🚀 দ্রুত ক্রিয়াকলাপ", 'quick_action_crop_desc': "আপনার মাটি এবং আবহাওয়ার উপর ভিত্তি করে উপযুক্ত ফসলের সুপারিশ পান।", 'quick_action_weather_desc': "আপনার এলাকার আবহাওয়া পরীক্ষা করুন এবং আপনার কৃষি কার্যক্রম পরিকল্পনা করুন।",
        'page_crop_guide': "ফসল গাইড", 'page_pest_prediction': "পোকা ভবিষ্যদ্বাণী", 'page_soil_type_prediction': "মাটির ধরন ভবিষ্যদ্বাণী",
        'crop_guide_title': "ফসল ফার্টিগেশন এবং সেচ গাইড", 'select_stage_label': "একটি বৃদ্ধি পর্যায় নির্বাচন করুন", 'guide_table_header': "নির্দেশনা বিবরণ",
        'file_not_found_error': "ফাইলটি খুঁজে পাওয়া যায়নি। অনুগ্রহ করে এটি সঠিক ফোল্ডারে আছে কিনা তা নিশ্চিত করুন।",
        'home_title': "কৃষিসখীতে স্বাগতম! 🌱", 'home_subtitle': "আধুনিক কৃষির জন্য আপনার AI-চালিত সহায়ক।",
        'home_intro': "*কৃষিসখী* কৃষকদের ডেটা-চালিত অন্তর্দৃষ্টি প্রদানের জন্য ডিজাইন করা হয়েছে.\nবিভিন্ন বৈশিষ্ট্য অ্যাক্সেস করতে সাইডবারের বিভাগগুলির মাধ্যমে নেভিগেট করুন:\n- *🌾 ফসল সুপারিশ:* মাটি এবং পরিবেশগত কারণের উপর ভিত্তি করে চাষ করার জন্য সেরা ফসলের জন্য সুপারিশ পান.\n- *📈 উৎপাদন ভবিষ্যদ্বাণী:* ঐতিহাসিক ডেটা এবং কৃষি পরামিতির উপর ভিত্তি করে আপনার ফসলের উৎপাদনের পূর্বাভাস করুন.\n- *🗓 ফসল ক্যালেন্ডার:* কেরালার বিভিন্ন ফসলের জন্য আদর্শ বীজ এবং কাটাই সময় দেখুন.\n- *🌦 আবহাওয়া রিপোর্ট:* আপনার অবস্থানের জন্য রিয়েল-টাইম আবহাওয়া আপডেট এবং পূর্বাভাস পান.\n- *🌿 পাতা রোগ ভবিষ্যদ্বাণী:* রোগগুলি তাড়াতাড়ি সনাক্ত করতে একটি পাতার ছবি আপলোড করুন.\n- *🌐 অনুবাদক:* বিভিন্ন ভারতীয় ভাষার মধ্যে কৃষি শব্দ এবং বাক্য অনুবাদ করুন.\n- *👥 কমিউনিটি ফোরাম:* অন্যান্য কৃষকদের সাথে সংযোগ করুন, প্রশ্ন জিজ্ঞাসা করুন এবং সমাধান ভাগ করুন।",
        'crop_rec_header': "🌾 ফসল সুপারিশ", 'crop_rec_intro': "ফসল সুপারিশ পেতে নিচের বিবরণগুলি প্রবেশ করান।",
        'nitrogen_label': "নাইট্রোজেন (N)", 'phosphorus_label': "ফসফরাস (P)", 'potassium_label': "পটাশিয়াম (K)", 'temperature_label': "তাপমাত্রা (°C)", 'humidity_label': "আর্দ্রতা (%)", 'ph_label': "pH মান", 'rainfall_label': "বৃষ্টিপাত (mm)",
        'recommend_crop_button': "ফসল সুপারিশ করুন", 'recommendation_success': "🎉 সুপারিশকৃত ফসল *{crop}*।",
        'crop_info_header': "{crop} সম্পর্কে আরও তথ্য", 'description_heading': "📖 বিবরণ", 'irrigation_heading': "💧 সেচ টিপস", 'conditions_heading': "☀️ আদর্শ অবস্থা", 'pests_heading': "🐛 সাধারণ কীড এবং রোগ",
        'yield_pred_header': "📈 ফসল উৎপাদন ভবিষ্যদ্বাণী", 'yield_pred_intro': "কেরালার একটি নির্দিষ্ট এলাকার জন্য ফসল উৎপাদন ভবিষ্যদ্বাণী করতে নিচের বিবরণগুলি পূরণ করুন।",
        'year_label': "বছর", 'crop_label': "ফসল", 'season_label': "মৌসুম", 'area_label': "এলাকা (হেক্টরে)",
        'predict_yield_button': "উৎপাদন ভবিষ্যদ্বাণী করুন", 'yield_prediction_success': "🌾 ভবিষ্যদ্বাণীকৃত ফসল উৎপাদন প্রায় *{yield_val:.2f} কেজি/হেক্টর*।",
        'market_price_header': "💰 আনুমানিক বাজার মূল্য", 'fetching_prices_spinner': "সাম্প্রতিক বাজার মূল্য আনা হচ্ছে...", 'market_price_label': "গড় বাজার মূল্য (প্রতি কুইন্টাল)", 'total_value_label': "মোট আনুমানিক মূল্য", 'profit_header': "💵 আনুমানিক লাভ", 'cost_per_ha_label': "কৃষি খরচ প্রতি হেক্টর (₹)", 'total_cost_label': "মোট কৃষি খরচ", 'total_profit_label': "মোট আনুমানিক লাভ", 'price_not_available': "{crop}-এর জন্য মূল্য ডেটা বর্তমানে অনুপলব্ধ।", 'price_api_error': "বর্তমানে বাজার মূল্য ডেটা পাওয়া যাচ্ছে না।",
        'crop_calendar_header': "🗓 ফসল বীজ এবং কাটাই ক্যালেন্ডার", 'select_crop_label': "একটি ফসল নির্বাচন করুন",
        'schedule_for_crop': "📅 {crop}-এর জন্য সময়সূচী", 'sowing_start_label': "বীজ শুরু", 'sowing_end_label': "বীজ শেষ", 'harvest_start_label': "কাটাই শুরু", 'harvest_end_label': "কাটাই শেষ", 'fertiliser_label': "সার (N:P:K কেজি/হেক্টর)", 'duration_label': "সময়কাল",
        'weather_report_header': "🌦 আবহাওয়া রিপোর্ট", 'enter_city_label': "আপনার শহরের নাম প্রবেশ করান", 'get_weather_button': "আবহাওয়া পান",
        'current_weather_in': "{location}-এ বর্তমান আবহাওয়া", 'temperature_metric': "তাপমাত্রা", 'feels_like_metric': "অনুভূত হয়", 'humidity_metric': "আর্দ্রতা", 'description_label': "বিবরণ:", 'wind_label': "বাতাস:", 'precipitation_label': "বৃষ্টিপাত:",
        'forecast_header': "🗓 3-দিনের পূর্বাভাস", 'avg_temp_label': "গড় তাপমাত্রা:", 'max_temp_label': "সর্বোচ্চ তাপমাত্রা:", 'min_temp_label': "সর্বনিম্ন তাপমাত্রা:", 'sunrise_label': "সূর্যোদয়:", 'sunset_label': "সূর্যাস্ত:", 'outlook_label': "দৃষ্টিভঙ্গি:",
        'error_parsing_weather': "আবহাওয়া ডেটা পার্স করা যায়নি। অনুগ্রহ করে শহরের নাম পরীক্ষা করে আবার চেষ্টা করুন। ত্রুটি: {e}", 'enter_city_warning': "⚠ অনুগ্রহ করে একটি শহরের নাম প্রবেশ করান।",
        'weather_fetch_error': "আবহাওয়া ডেটা পাওয়া যায়নি। অনুগ্রহ করে আপনার নেটওয়ার্ক সংযোগ বা শহরের নাম পরীক্ষা করে আবার চেষ্টা করুন।", 'weather_api_key_error': "API কী ত্রুটি: {message}. অনুগ্রহ করে secrets.toml-এ আপনার OpenWeatherMap API কী পরীক্ষা করুন। নতুন কীগুলি সক্রিয় হতে কয়েক ঘণ্টা সময় নিতে পারে।", 'weather_city_not_found_error': "শহর খুঁজে পাওয়া যায়নি ত্রুটি: {message}. অনুগ্রহ করে শহরের নামের বানান পরীক্ষা করুন।", 'weather_api_generic_error': "API ত্রুটি ({cod}): {message}",
        'translator_header': "🌐 কৃষক ভাষা অনুবাদক", 'translator_intro': "✍ অনুবাদ করার জন্য টেক্সট প্রবেশ করান", 'translate_button': "🔄 অনুবাদ করুন", 'translation_failed_error': "❌ অনুবাদ ব্যর্থ হয়েছে: {e}", 'enter_text_warning': "⚠ অনুগ্রহ করে কিছু টেক্সট প্রবেশ করান।",
        'ai_chatbot_header': "🤖 AI চ্যাটবট", 'ai_chatbot_intro': "এই বৈশিষ্ট্যটি শীঘ্রই আসছে! আপনার কৃষি প্রশ্নের সাহায্য করার জন্য একটি শক্তিশালী AI চ্যাটবটের জন্য অপেক্ষা করুন।",
        'languages_supported_label': "সমর্থিত ভাষা", 'farmers_helped_label': "সাহায্যপ্রাপ্ত কৃষক", 'crops_covered_label': "আচ্ছাদিত ফসল", 'recommendations_given_label': "AI সুপারিশ",
        'leaf_disease_header': "🌿 পাতা রোগ ভবিষ্যদ্বাণী", 'leaf_disease_intro': "রোগগুলি পরীক্ষা করতে একটি গাছের পাতার ছবি আপলোড করুন।", 'model_not_found_warning': "⚠ রোগ ভবিষ্যদ্বাণী মডেল খুঁজে পাওয়া যায়নি। অনুগ্রহ করে প্রশাসকের সাথে যোগাযোগ করুন।",
        'uploader_label': "একটি পাতার ছবি চয়ন করুন...", 'uploaded_image_caption': "আপলোড করা ছবি।", 'predict_disease_button': "🔍 রোগ ভবিষ্যদ্বাণী করুন",
        'predicting_spinner': "পাতা বিশ্লেষণ করা হচ্ছে...", 'prediction_result': "✅ ভবিষ্যদ্বাণী: *{disease}*",
        'remedy_info_header': "{disease}-এর জন্য চিকিৎসা এবং ব্যবস্থাপনা", 'disease_description_heading': "ℹ️ রোগ সম্পর্কে", 'organic_remedies_heading': "🌿 জৈব প্রতিকার", 'chemical_remedies_heading': "🧪 রাসায়নিক প্রতিকার", 'healthy_plant_message': "খুশির খবর! আপনার গাছটি সুস্থ বলে মনে হচ্ছে। ভাল কৃষি অনুশীলন চালিয়ে যান।",
        'soil_prediction_header': "🏞️ মাটির ধরন ভবিষ্যদ্বাণী", 'soil_prediction_intro': "মাটির ধরন সনাক্ত করতে এবং এর বৈশিষ্ট্য সম্পর্কে জানতে একটি মাটির ছবি আপলোড করুন।", 'upload_soil_image_label': "একটি মাটির ছবি চয়ন করুন...",
        'predict_soil_button': "🔍 মাটির ধরন ভবিষ্যদ্বাণী করুন", 'analyzing_soil_spinner': "মাটি বিশ্লেষণ করা হচ্ছে...", 'soil_prediction_result': "✅ ভবিষ্যদ্বাণীকৃত মাটির ধরন: *{soil_type}*", 'soil_info_header': "{soil_type} সম্পর্কে তথ্য",
        'pest_prediction_header': "ছবি থেকে পোকা সনাক্তকরণ", 'upload_image_pest': "পোকা বা প্রভাবিত পাতার একটি ছবি আপলোড করুন", 'prediction_pest': "ভবিষ্যদ্বাণীকৃত পোকা / পতঙ্গ", 'model_loading_error': "পোকা ভবিষ্যদ্বাণী মডেল লোড করা যায়নি। অনুগ্রহ করে ইন্টারনেট সংযোগ পরীক্ষা করুন।",
        'remedy_info_header_pest': "{pest}-এর জন্য চিকিৎসা এবং ব্যবস্থাপনা", 'pest_description_heading': "ℹ️ পোকা সম্পর্কে", 'no_remedy_info': "{pest}-এর জন্য প্রতিকার তথ্য এখনও উপলব্ধ নয়।",
        'feedback_header': "প্রতিক্রিয়া", 'rating_label': "আপনার অভিজ্ঞতা রেট করুন:", 'submit_button': "জমা দিন", 'feedback_success': "আপনার প্রতিক্রিয়ার জন্য ধন্যবাদ!",
        'contact_header': "যোগাযোগ করুন",
        'login_page_title': "কৃষিসখীতে স্বাগতম 🌱", 'login_tab': "লগইন", 'signup_tab': "সাইন আপ",
        'username_label': "ব্যবহারকারীর নাম", 'password_label': "পাসওয়ার্ড", 'login_button': "লগইন", 'signup_button': "সাইন আপ", 'logout_button': "লগ আউট",
        'login_error': "ভুল ব্যবহারকারীর নাম বা পাসওয়ার্ড।", 'signup_success': "নিবন্ধন সফল! অনুগ্রহ করে লগইন করুন।", 'signup_error_exists': "ব্যবহারকারীর নাম ইতিমধ্যে বিদ্যমান। অনুগ্রহ করে অন্য একটি চয়ন করুন।", 'signup_error_general': "নিবন্ধনের সময় একটি ত্রুটি ঘটেছে।",
        'welcome_back': "ফিরে আসার জন্য স্বাগতম, {username}!",
        'profile_header': "👤 আমার প্রোফাইল", 'profile_subheader': "আপনার অ্যাকাউন্টের বিবরণ এবং সেটিংস পরিচালনা করুন।",
        'full_name_label': "পুরো নাম", 'contact_number_label': "যোগাযোগ নম্বর", 'new_username_label': "নতুন ব্যবহারকারীর নাম", 'new_password_label': "নতুন পাসওয়ার্ড", 'confirm_password_label': "নতুন পাসওয়ার্ড নিশ্চিত করুন", 'profile_pic_label': "প্রোফাইল ছবি",
        'update_profile_button': "প্রোফাইল আপডেট করুন", 'profile_update_success': "✅ প্রোফাইল সফলভাবে আপডেট হয়েছে!", 'profile_update_error': "❌ প্রোফাইল আপডেট করা যায়নি। ব্যবহারকারীর নাম ইতিমধ্যে বিদ্যমান হতে পারে।", 'password_mismatch_error': "পাসওয়ার্ড মিলছে না।",
        'danger_zone_header': "🚨 বিপজ্জনক অঞ্চল", 'delete_account_button': "আমার অ্যাকাউন্ট মুছুন", 'delete_account_warning': "এই ক্রিয়াটি অপরিবর্তনীয়। আপনার সমস্ত ডেটা স্থায়ীভাবে মুছে ফেলা হবে।", 'delete_confirmation_label': "নিশ্চিত করতে, অনুগ্রহ করে নিচের বাক্সে `DELETE` টাইপ করুন।",
        'delete_account_confirm_button': "হ্যাঁ, আমি আমার অ্যাকাউন্ট মুছতে চাই", 'account_deleted_success': "আপনার অ্যাকাউন্ট সফলভাবে মুছে ফেলা হয়েছে।", 'incorrect_delete_confirmation': "ভুল নিশ্চিতকরণ টেক্সট।",
        'logs_header': "আমার কৃষি লগ", 'logs_intro': "এখানে আপনার দৈনন্দিন কার্যক্রম, পর্যবেক্ষণ এবং নোটস রেকর্ড করুন।", 'new_log_label': "নতুন লগ এন্ট্রি", 'save_log_button': "লগ সেভ করুন", 'log_saved_success': "লগ এন্ট্রি সফলভাবে সেভ হয়েছে!", 'past_logs_header': "📖 পূর্ববর্তী এন্ট্রি", 'no_logs_message': "আপনার এখনও কোন লগ এন্ট্রি নেই।", 'delete_log_button': "মুছুন", 'log_deleted_success': "লগ মুছে ফেলা হয়েছে!", 'record_log_prompt': "একটি ভয়েস নোট সংযুক্ত করুন:", 'voice_note_recorded': "🎤 ভয়েস নোট রেকর্ড করা হয়েছে এবং আপনার লগের সাথে সেভ করার জন্য প্রস্তুত।", 'attached_voice_note': "সংযুক্ত ভয়েস নোট:",
        'upload_media_label': "ছবি বা ভিডিও সংযুক্ত করুন:", 'attached_images': "সংযুক্ত ছবি:", 'attached_videos': "সংযুক্ত ভিডিও:",
        'forum_header': "👥 কমিউনিটি ফোরাম", 'forum_intro': "অন্যান্য কৃষকদের সাথে সংযোগ করুন। প্রশ্ন জিজ্ঞাসা করুন, সমাধান ভাগ করুন এবং একসাথে বাড়ুন।",
        'create_post_header': "একটি নতুন পোস্ট তৈরি করুন", 'post_title_label': "শিরোনাম", 'post_title_placeholder': "একটি স্পষ্ট এবং সংক্ষিপ্ত শিরোনাম প্রবেশ করান", 'post_content_label': "আপনার প্রশ্ন বা টিপ", 'post_content_placeholder': "আপনার সমস্যা তপশীলভাবে বর্ণনা করা বা আপনার জ্ঞান ভাগ করুন...",
        'upload_image_label': "একটি ছবি আপলোড করুন (ঐচ্ছিক)", 'submit_post_button': "পোস্ট জমা দিন", 'post_success_message': "✅ আপনার পোস্ট প্রকাশিত হয়েছে!",
        'all_posts_header': "সমস্ত পোস্ট", 'posted_by': "{username} দ্বারা {date}-এ পোস্ট করা হয়েছে", 'like_button': "👍 লাইক ({count})", 'reply_button': "💬 উত্তর", 'replies_header': "উত্তর", 'add_reply_placeholder': "একটি উত্তর লিখুন...", 'submit_reply_button': "উত্তর পোস্ট করুন",
        'no_posts_message': "এখনও কোন পোস্ট নেই। কথোপকথন শুরু করার প্রথম ব্যক্তি হন!", 'view_replies': "{count} উত্তর দেখুন", 'hide_replies': "উত্তর লুকান",
        'soil_characteristics_heading': "🔬 বৈশিষ্ট্য", 'soil_suitable_crops_heading': "🌱 উপযুক্ত ফসল", 'soil_model_not_found_warning': "⚠ মাটি ভবিষ্যদ্বাণী মডেল খুঁজে পাওয়া যায়নি। অনুগ্রহ করে প্রশাসকের সাথে যোগাযোগ করুন।",
        'stage_seedling': 'চারা', 'stage_vegetative': 'উদ্ভিদ', 'stage_flowering': 'ফুল', 'stage_fruiting': 'ফল', 'stage_maturity': 'পরিপক্কতা',
        'column_days_range': 'দিনের পরিসীমা', 'column_fertilizer': 'সার', 'column_dosage': 'ডোজ (কেজি/একর)', 'column_frequency': 'ফ্রিকোয়েন্সি', 'column_irrigation_depth': 'সেচ গভীরতা / পরিমাণ', 'column_irrigation_frequency': 'সেচ ফ্রিকোয়েন্সি',
    },
    'pa': {
        'app_title': "ਕ੍ਰਿਸ਼ੀਸਖੀ", 'go_to': "ਜਾਓ", 'go_button': "ਜਾਓ", 'back_to_home': "ਘਰ ਵਾਪਸ ਜਾਓ", 'language_selector_label': "ਭਾਸ਼ਾ",
        'page_home': "ਘਰ", 'page_crop_recommendation': "ਫਸਲ ਸਿਫਾਰਸ਼", 'page_yield_prediction': "ਉਪਜ ਭਵਿੱਖਬਾਣੀ", 'page_crop_calendar': "ਫਸਲ ਕੈਲੰਡਰ", 'page_weather_report': "ਮੌਸਮ ਰਿਪੋਰਟ", 'page_leaf_disease_prediction': "ਪੱਤਾ ਬਿਮਾਰੀ ਭਵਿੱਖਬਾਣੀ", 'page_translator': "ਅਨੁਵਾਦਕ", 'page_profile': "ਪ੍ਰੋਫਾਈਲ", 'page_logs': "ਲੌਗ", 'page_community_forum': "ਕਮਿਊਨਿਟੀ ਫੋਰਮ",
        'quick_actions_title': "🚀 ਤੇਜ਼ ਕਾਰਵਾਈਆਂ", 'quick_action_crop_desc': "ਆਪਣੀ ਮਿੱਟੀ ਅਤੇ ਮੌਸਮ ਦੇ ਆਧਾਰ 'ਤੇ ਢੁਕਵੀਆਂ ਫਸਲ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰੋ।", 'quick_action_weather_desc': "ਆਪਣੇ ਖੇਤਰ ਦਾ ਮੌਸਮ ਜਾਂਚੋ ਅਤੇ ਆਪਣੀਆਂ ਕ੍ਰਿਸ਼ੀ ਗਤੀਵਿਧੀਆਂ ਦੀ ਯੋਜਨਾ ਬਣਾਓ।",
        'page_crop_guide': "ਫਸਲ ਗਾਈਡ", 'page_pest_prediction': "ਕੀੜਾ ਭਵਿੱਖਬਾਣੀ", 'page_soil_type_prediction': "ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਭਵਿੱਖਬਾਣੀ",
        'crop_guide_title': "ਫਸਲ ਫਰਟੀਗੇਸ਼ਨ ਅਤੇ ਸਿੰਚਾਈ ਗਾਈਡ", 'select_stage_label': "ਇੱਕ ਵਿਕਾਸ ਪੜਾਅ ਚੁਣੋ", 'guide_table_header': "ਮਾਰਗਦਰਸ਼ਨ ਵੇਰਵਾ",
        'file_not_found_error': "ਫਾਈਲ ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਇਹ ਸੁਨਿਸ਼ਚਿਤ ਕਰੋ ਕਿ ਇਹ ਸਹੀ ਫੋਲਡਰ ਵਿੱਚ ਹੈ।",
        'home_title': "ਕ੍ਰਿਸ਼ੀਸਖੀ ਵਿੱਚ ਤੁਹਾਡਾ ਸੁਆਗਤ ਹੈ! 🌱", 'home_subtitle': "ਆਧੁਨਿਕ ਕ੍ਰਿਸ਼ੀ ਲਈ ਤੁਹਾਡਾ AI-ਚਲਿਤ ਸਹਾਇਕ।",
        'home_intro': "*ਕ੍ਰਿਸ਼ੀਸਖੀ* ਕਿਸਾਨਾਂ ਨੂੰ ਡੇਟਾ-ਚਲਿਤ ਇਨਸਾਈਟਸ ਪ੍ਰਦਾਨ ਕਰਨ ਲਈ ਡਿਜ਼ਾਈਨ ਕੀਤਾ ਗਿਆ ਹੈ.\nਵੱਖ-ਵੱਖ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ ਤੱਕ ਪਹੁੰਚ ਕਰਨ ਲਈ ਸਾਈਡਬਾਰ ਵਿੱਚ ਭਾਗਾਂ ਦੁਆਰਾ ਨੈਵੀਗੇਟ ਕਰੋ:\n- *🌾 ਫਸਲ ਸਿਫਾਰਸ਼:* ਮਿੱਟੀ ਅਤੇ ਵਾਤਾਵਰਣਿਕ ਕਾਰਕਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਬੀਜਣ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲਾਂ ਲਈ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰੋ.\n- *📈 ਉਪਜ ਭਵਿੱਖਬਾਣੀ:* ਇਤਿਹਾਸਿਕ ਡੇਟਾ ਅਤੇ ਕ੍ਰਿਸ਼ੀ ਪੈਰਾਮੀਟਰਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਆਪਣੀ ਫਸਲ ਦੀ ਉਪਜ ਦਾ ਅਨੁਮਾਨ ਲਗਾਓ.\n- *🗓 ਫਸਲ ਕੈਲੰਡਰ:* ਕੇਰਲਾ ਵਿੱਚ ਵੱਖ-ਵੱਖ ਫਸਲਾਂ ਲਈ ਆਦਰਸ਼ ਬੀਜਣ ਅਤੇ ਕਟਾਈ ਦੇ ਸਮੇਂ ਦੇਖੋ.\n- *🌦 ਮੌਸਮ ਰਿਪੋਰਟ:* ਆਪਣੇ ਸਥਾਨ ਲਈ ਰੀਅਲ-ਟਾਈਮ ਮੌਸਮ ਅਪਡੇਟ ਅਤੇ ਅਨੁਮਾਨ ਪ੍ਰਾਪਤ ਕਰੋ.\n- *🌿 ਪੱਤਾ ਬਿਮਾਰੀ ਭਵਿੱਖਬਾਣੀ:* ਬਿਮਾਰੀਆਂ ਨੂੰ ਜਲਦੀ ਪਛਾਣਨ ਲਈ ਇੱਕ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ.\n- *🌐 ਅਨੁਵਾਦਕ:* ਵੱਖ-ਵੱਖ ਭਾਸ਼ਾਵਾਂ ਵਿੱਚ ਕ੍ਰਿਸ਼ੀ ਸ਼ਬਦਾਂ ਅਤੇ ਵਾਕਾਂ ਦਾ ਅਨੁਵਾਦ ਕਰੋ.\n- *👥 ਕਮਿਊਨਿਟੀ ਫੋਰਮ:* ਹੋਰ ਕਿਸਾਨਾਂ ਨਾਲ ਜੁੜੋ, ਸਵਾਲ ਪੁੱਛੋ ਅਤੇ ਹੱਲ ਸਾਂਝੇ ਕਰੋ।",
        'crop_rec_header': "🌾 ਫਸਲ ਸਿਫਾਰਸ਼", 'crop_rec_intro': "ਫਸਲ ਸਿਫਾਰਸ਼ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਹੇਠਾਂ ਦਿੱਤੇ ਵੇਰਵੇ ਦਾਖਲ ਕਰੋ।",
        'nitrogen_label': "ਨਾਈਟ੍ਰੋਜਨ (N)", 'phosphorus_label': "ਫਾਸਫੋਰਸ (P)", 'potassium_label': "ਪੋਟਾਸ਼ੀਅਮ (K)", 'temperature_label': "ਤਾਪਮਾਨ (°C)", 'humidity_label': "ਨਮੀ (%)", 'ph_label': "pH ਮੁੱਲ", 'rainfall_label': "ਬਾਰਸ਼ (mm)",
        'recommend_crop_button': "ਫਸਲ ਸਿਫਾਰਸ਼ ਕਰੋ", 'recommendation_success': "🎉 ਸਿਫਾਰਸ਼ੀ ਫਸਲ *{crop}* ਹੈ।",
        'crop_info_header': "{crop} ਬਾਰੇ ਹੋਰ ਜਾਣਕਾਰੀ", 'description_heading': "📖 ਵੇਰਵਾ", 'irrigation_heading': "💧 ਸਿੰਚਾਈ ਟਿਪਸ", 'conditions_heading': "☀️ ਆਦਰਸ਼ ਹਾਲਤਾਂ", 'pests_heading': "🐛 ਆਮ ਕੀੜੇ ਅਤੇ ਬਿਮਾਰੀਆਂ",
        'yield_pred_header': "📈 ਫਸਲ ਉਪਜ ਭਵਿੱਖਬਾਣੀ", 'yield_pred_intro': "ਕੇਰਲਾ ਦੇ ਇੱਕ ਖਾਸ ਖੇਤਰ ਲਈ ਫਸਲ ਉਪਜ ਦਾ ਅਨੁਮਾਨ ਲਗਾਉਣ ਲਈ ਹੇਠਾਂ ਦਿੱਤੇ ਵੇਰਵੇ ਭਰੋ।",
        'year_label': "ਸਾਲ", 'crop_label': "ਫਸਲ", 'season_label': "ਮੌਸਮ", 'area_label': "ਖੇਤਰ (ਹੈਕਟੇਅਰ ਵਿੱਚ)",
        'predict_yield_button': "ਉਪਜ ਦਾ ਅਨੁਮਾਨ ਲਗਾਓ", 'yield_prediction_success': "🌾 ਭਵਿੱਖਬਾਣੀ ਕੀਤੀ ਫਸਲ ਉਪਜ ਲਗਭਗ *{yield_val:.2f} ਕਿਲੋ/ਹੈਕਟੇਅਰ* ਹੈ।",
        'market_price_header': "💰 ਅਨੁਮਾਨਿਤ ਮਾਰਕੀਟ ਮੁੱਲ", 'fetching_prices_spinner': "ਤਾਜ਼ਾ ਮਾਰਕੀਟ ਕੀਮਤਾਂ ਲਿਆਈਆਂ ਜਾ ਰਹੀਆਂ ਹਨ...", 'market_price_label': "ਔਸਤ ਮਾਰਕੀਟ ਕੀਮਤ (ਪ੍ਰਤੀ ਕੁਇੰਟਲ)", 'total_value_label': "ਕੁੱਲ ਅਨੁਮਾਨਿਤ ਮੁੱਲ", 'profit_header': "💵 ਅਨੁਮਾਨਿਤ ਲਾਭ", 'cost_per_ha_label': "ਕ੍ਰਿਸ਼ੀ ਖਰਚ ਪ੍ਰਤੀ ਹੈਕਟੇਅਰ (₹)", 'total_cost_label': "ਕੁੱਲ ਕ੍ਰਿਸ਼ੀ ਖਰਚ", 'total_profit_label': "ਕੁੱਲ ਅਨੁਮਾਨਿਤ ਲਾਭ", 'price_not_available': "{crop} ਲਈ ਕੀਮਤ ਡੇਟਾ ਇਸ ਵੇਲੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।", 'price_api_error': "ਇਸ ਵੇਲੇ ਮਾਰਕੀਟ ਕੀਮਤ ਡੇਟਾ ਪ੍ਰਾਪਤ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਦਾ।",
        'crop_calendar_header': "🗓 ਫਸਲ ਬੀਜਣ ਅਤੇ ਕਟਾਈ ਕੈਲੰਡਰ", 'select_crop_label': "ਇੱਕ ਫਸਲ ਚੁਣੋ",
        'schedule_for_crop': "📅 {crop} ਲਈ ਸਮਾਂ-ਸਾਰਣੀ", 'sowing_start_label': "ਬੀਜਣ ਸ਼ੁਰੂ", 'sowing_end_label': "ਬੀਜਣ ਖਤਮ", 'harvest_start_label': "ਕਟਾਈ ਸ਼ੁਰੂ", 'harvest_end_label': "ਕਟਾਈ ਖਤਮ", 'fertiliser_label': "ਖਾਦ (N:P:K ਕਿਲੋ/ਹੈਕਟੇਅਰ)", 'duration_label': "ਅਵਧੀ",
        'weather_report_header': "🌦 ਮੌਸਮ ਰਿਪੋਰਟ", 'enter_city_label': "ਆਪਣੇ ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਦਾਖਲ ਕਰੋ", 'get_weather_button': "ਮੌਸਮ ਪ੍ਰਾਪਤ ਕਰੋ",
        'current_weather_in': "{location} ਵਿੱਚ ਮੌਜੂਦਾ ਮੌਸਮ", 'temperature_metric': "ਤਾਪਮਾਨ", 'feels_like_metric': "ਅਹਿਸਾਸ ਹੁੰਦਾ ਹੈ", 'humidity_metric': "ਨਮੀ", 'description_label': "ਵੇਰਵਾ:", 'wind_label': "ਹਵਾ:", 'precipitation_label': "ਬਾਰਸ਼:",
        'forecast_header': "🗓 3-ਦਿਨ ਦਾ ਅਨੁਮਾਨ", 'avg_temp_label': "ਔਸਤ ਤਾਪਮਾਨ:", 'max_temp_label': "ਅਧਿਕਤਮ ਤਾਪਮਾਨ:", 'min_temp_label': "ਨਿਊਨਤਮ ਤਾਪਮਾਨ:", 'sunrise_label': "ਸੂਰਜ ਉੱਗਣਾ:", 'sunset_label': "ਸੂਰਜ ਡੁੱਬਣਾ:", 'outlook_label': "ਨਜ਼ਰੀਆ:",
        'error_parsing_weather': "ਮੌਸਮ ਡੇਟਾ ਪਾਰਸ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਜਾਂਚ ਕਰੋ ਅਤੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ। ਗਲਤੀ: {e}", 'enter_city_warning': "⚠ ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਦਾਖਲ ਕਰੋ।",
        'weather_fetch_error': "ਮੌਸਮ ਡੇਟਾ ਪ੍ਰਾਪਤ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਨੈਟਵਰਕ ਕਨੈਕਸ਼ਨ ਜਾਂ ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਜਾਂਚ ਕਰੋ ਅਤੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।", 'weather_api_key_error': "API ਕੁੰਜੀ ਗਲਤੀ: {message}. ਕਿਰਪਾ ਕਰਕੇ secrets.toml ਵਿੱਚ ਆਪਣੀ OpenWeatherMap API ਕੁੰਜੀ ਜਾਂਚ ਕਰੋ। ਨਵੀਆਂ ਕੁੰਜੀਆਂ ਸਰਗਰਮ ਹੋਣ ਵਿੱਚ ਕੁਝ ਘੰਟੇ ਲਗ ਸਕਦੇ ਹਨ।", 'weather_city_not_found_error': "ਸ਼ਹਿਰ ਨਹੀਂ ਮਿਲਿਆ ਗਲਤੀ: {message}. ਕਿਰਪਾ ਕਰਕੇ ਸ਼ਹਿਰ ਦੇ ਨਾਮ ਦੀ ਸ਼ਬਦ-ਜੋੜ ਜਾਂਚ ਕਰੋ।", 'weather_api_generic_error': "API ਗਲਤੀ ({cod}): {message}",
        'translator_header': "🌐 ਕਿਸਾਨ ਭਾਸ਼ਾ ਅਨੁਵਾਦਕ", 'translator_intro': "✍ ਅਨੁਵਾਦ ਕਰਨ ਲਈ ਟੈਕਸਟ ਦਾਖਲ ਕਰੋ", 'translate_button': "🔄 ਅਨੁਵਾਦ ਕਰੋ", 'translation_failed_error': "❌ ਅਨੁਵਾਦ ਅਸਫਲ: {e}", 'enter_text_warning': "⚠ ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਟੈਕਸਟ ਦਾਖਲ ਕਰੋ।",
        'ai_chatbot_header': "🤖 AI ਚੈਟਬੋਟ", 'ai_chatbot_intro': "ਇਹ ਵਿਸ਼ੇਸ਼ਤਾ ਜਲਦੀ ਆ ਰਹੀ ਹੈ! ਆਪਣੇ ਕ੍ਰਿਸ਼ੀ ਸਵਾਲਾਂ ਵਿੱਚ ਮਦਦ ਕਰਨ ਲਈ ਇੱਕ ਸ਼ਕਤੀਸ਼ਾਲੀ AI ਚੈਟਬੋਟ ਦੀ ਉਡੀਕ ਕਰੋ।",
        'languages_supported_label': "ਸਮਰਥਿਤ ਭਾਸ਼ਾਵਾਂ", 'farmers_helped_label': "ਮਦਦ ਕੀਤੇ ਕਿਸਾਨ", 'crops_covered_label': "ਕਵਰ ਕੀਤੀਆਂ ਫਸਲਾਂ", 'recommendations_given_label': "AI ਸਿਫਾਰਸ਼ਾਂ",
        'leaf_disease_header': "🌿 ਪੱਤਾ ਬਿਮਾਰੀ ਭਵਿੱਖਬਾਣੀ", 'leaf_disease_intro': "ਬਿਮਾਰੀਆਂ ਦੀ ਜਾਂਚ ਕਰਨ ਲਈ ਇੱਕ ਬੂਟੇ ਦੇ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ।", 'model_not_found_warning': "⚠ ਬਿਮਾਰੀ ਭਵਿੱਖਬਾਣੀ ਮਾਡਲ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਪ੍ਰਸ਼ਾਸਕ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
        'uploader_label': "ਇੱਕ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਚੁਣੋ...", 'uploaded_image_caption': "ਅਪਲੋਡ ਕੀਤੀ ਤਸਵੀਰ।", 'predict_disease_button': "🔍 ਬਿਮਾਰੀ ਦਾ ਅਨੁਮਾਨ ਲਗਾਓ",
        'predicting_spinner': "ਪੱਤਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...", 'prediction_result': "✅ ਭਵਿੱਖਬਾਣੀ: *{disease}*",
        'remedy_info_header': "{disease} ਲਈ ਇਲਾਜ ਅਤੇ ਪ੍ਰਬੰਧਨ", 'disease_description_heading': "ℹ️ ਬਿਮਾਰੀ ਬਾਰੇ", 'organic_remedies_heading': "🌿 ਜੈਵਿਕ ਉਪਚਾਰ", 'chemical_remedies_heading': "🧪 ਰਸਾਇਣਕ ਉਪਚਾਰ", 'healthy_plant_message': "ਖੁਸ਼ੀ ਦੀ ਖ਼ਬਰ! ਤੁਹਾਡਾ ਬੂਟਾ ਸਿਹਤਮੰਦ ਲਗ ਰਿਹਾ ਹੈ। ਵਧੀਆ ਕ੍ਰਿਸ਼ੀ ਅਭਿਆਸ ਜਾਰੀ ਰੱਖੋ।",
        'soil_prediction_header': "🏞️ ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਭਵਿੱਖਬਾਣੀ", 'soil_prediction_intro': "ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਪਛਾਣ ਕਰਨ ਅਤੇ ਇਸਦੇ ਗੁਣਾਂ ਬਾਰੇ ਜਾਣਨ ਲਈ ਇੱਕ ਮਿੱਟੀ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ।", 'upload_soil_image_label': "ਇੱਕ ਮਿੱਟੀ ਦੀ ਤਸਵੀਰ ਚੁਣੋ...",
        'predict_soil_button': "🔍 ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਦਾ ਅਨੁਮਾਨ ਲਗਾਓ", 'analyzing_soil_spinner': "ਮਿੱਟੀ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ...", 'soil_prediction_result': "✅ ਭਵਿੱਖਬਾਣੀ ਕੀਤੀ ਮਿੱਟੀ ਦੀ ਕਿਸਮ: *{soil_type}*", 'soil_info_header': "{soil_type} ਬਾਰੇ ਜਾਣਕਾਰੀ",
        'pest_prediction_header': "ਤਸਵੀਰ ਤੋਂ ਕੀੜਾ ਪਛਾਣ", 'upload_image_pest': "ਕੀੜੇ ਜਾਂ ਪ੍ਰਭਾਵਿਤ ਪੱਤੇ ਦੀ ਇੱਕ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ", 'prediction_pest': "ਭਵਿੱਖਬਾਣੀ ਕੀਤਾ ਕੀੜਾ / ਕੀੜਾ", 'model_loading_error': "ਕੀੜਾ ਭਵਿੱਖਬਾਣੀ ਮਾਡਲ ਲੋਡ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਦਾ। ਕਿਰਪਾ ਕਰਕੇ ਇੰਟਰਨੈੱਟ ਕਨੈਕਸ਼ਨ ਜਾਂਚ ਕਰੋ।",
        'remedy_info_header_pest': "{pest} ਲਈ ਇਲਾਜ ਅਤੇ ਪ੍ਰਬੰਧਨ", 'pest_description_heading': "ℹ️ ਕੀੜੇ ਬਾਰੇ", 'no_remedy_info': "{pest} ਲਈ ਉਪਚਾਰ ਜਾਣਕਾਰੀ ਅਜੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        'feedback_header': "ਪ੍ਰਤੀਕ੍ਰਿਆ", 'rating_label': "ਆਪਣੇ ਅਨੁਭਵ ਦਾ ਮੁਲਾਂਕਣ ਕਰੋ:", 'submit_button': "ਜਮ੍ਹਾਂ ਕਰੋ", 'feedback_success': "ਤੁਹਾਡੀ ਪ੍ਰਤੀਕ੍ਰਿਆ ਲਈ ਧੰਨਵਾਦ!",
        'contact_header': "ਸੰਪਰਕ ਕਰੋ",
        'login_page_title': "ਕ੍ਰਿਸ਼ੀਸਖੀ ਵਿੱਚ ਸੁਆਗਤ ਹੈ 🌱", 'login_tab': "ਲੌਗਇਨ", 'signup_tab': "ਸਾਈਨ ਅੱਪ",
        'username_label': "ਯੂਜ਼ਰਨੇਮ", 'password_label': "ਪਾਸਵਰਡ", 'login_button': "ਲੌਗਇਨ", 'signup_button': "ਸਾਈਨ ਅੱਪ", 'logout_button': "ਲੌਗ ਆਉਟ",
        'login_error': "ਗਲਤ ਯੂਜ਼ਰਨੇਮ ਜਾਂ ਪਾਸਵਰਡ।", 'signup_success': "ਰਜਿਸਟ੍ਰੇਸ਼ਨ ਸਫਲ! ਕਿਰਪਾ ਕਰਕੇ ਲੌਗਇਨ ਕਰੋ।", 'signup_error_exists': "ਯੂਜ਼ਰਨੇਮ ਪਹਿਲਾਂ ਹੀ ਮੌਜੂਦ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੋਈ ਹੋਰ ਚੁਣੋ।", 'signup_error_general': "ਰਜਿਸਟ੍ਰੇਸ਼ਨ ਦੌਰਾਨ ਇੱਕ ਗਲਤੀ ਹੋਈ।",
        'welcome_back': "ਵਾਪਸ ਆਉਣ ਲਈ ਸੁਆਗਤ ਹੈ, {username}!",
        'profile_header': "👤 ਮੇਰਾ ਪ੍ਰੋਫਾਈਲ", 'profile_subheader': "ਆਪਣੇ ਖਾਤੇ ਦੇ ਵੇਰਵੇ ਅਤੇ ਸੈਟਿੰਗਾਂ ਦਾ ਪ੍ਰਬੰਧਨ ਕਰੋ।",
        'full_name_label': "ਪੂਰਾ ਨਾਮ", 'contact_number_label': "ਸੰਪਰਕ ਨੰਬਰ", 'new_username_label': "ਨਵਾਂ ਯੂਜ਼ਰਨੇਮ", 'new_password_label': "ਨਵਾਂ ਪਾਸਵਰਡ", 'confirm_password_label': "ਨਵੇਂ ਪਾਸਵਰਡ ਦੀ ਪੁਸ਼ਟੀ ਕਰੋ", 'profile_pic_label': "ਪ੍ਰੋਫਾਈਲ ਤਸਵੀਰ",
        'update_profile_button': "ਪ੍ਰੋਫਾਈਲ ਅਪਡੇਟ ਕਰੋ", 'profile_update_success': "✅ ਪ੍ਰੋਫਾਈਲ ਸਫਲਤਾਪੂਰਵਕ ਅਪਡੇਟ ਹੋਇਆ!", 'profile_update_error': "❌ ਪ੍ਰੋਫਾਈਲ ਅਪਡੇਟ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਿਆ। ਯੂਜ਼ਰਨੇਮ ਪਹਿਲਾਂ ਹੀ ਮੌਜੂਦ ਹੋ ਸਕਦਾ ਹੈ।", 'password_mismatch_error': "ਪਾਸਵਰਡ ਮੇਲ ਨਹੀਂ ਖਾਂਦੇ।",
        'danger_zone_header': "🚨 ਖ਼ਤਰਨਾਕ ਖੇਤਰ", 'delete_account_button': "ਮੇਰਾ ਖਾਤਾ ਮਿਟਾਓ", 'delete_account_warning': "ਇਹ ਕਾਰਵਾਈ ਅਟੱਲ ਹੈ। ਤੁਹਾਡਾ ਸਾਰਾ ਡੇਟਾ ਸਥਾਈ ਤੌਰ 'ਤੇ ਮਿਟਾ ਦਿੱਤਾ ਜਾਵੇਗਾ।", 'delete_confirmation_label': "ਪੁਸ਼ਟੀ ਕਰਨ ਲਈ, ਕਿਰਪਾ ਕਰਕੇ ਹੇਠਾਂ ਦਿੱਤੇ ਬਕਸੇ ਵਿੱਚ `DELETE` ਟਾਈਪ ਕਰੋ।",
        'delete_account_confirm_button': "ਹਾਂ, ਮੈਂ ਆਪਣਾ ਖਾਤਾ ਮਿਟਾਉਣਾ ਚਾਹੁੰਦਾ ਹਾਂ", 'account_deleted_success': "ਤੁਹਾਡਾ ਖਾਤਾ ਸਫਲਤਾਪੂਰਵਕ ਮਿਟਾ ਦਿੱਤਾ ਗਿਆ ਹੈ।", 'incorrect_delete_confirmation': "ਗਲਤ ਪੁਸ਼ਟੀਕਰਨ ਟੈਕਸਟ।",
        'logs_header': "ਮੇਰੇ ਕ੍ਰਿਸ਼ੀ ਲੌਗ", 'logs_intro': "ਇੱਥੇ ਆਪਣੀਆਂ ਦੈਨਿਕ ਗਤੀਵਿਧੀਆਂ, ਪ੍ਰਤੀਭਾਸ਼ਾ ਅਤੇ ਨੋਟਸ ਰਿਕਾਰਡ ਕਰੋ।", 'new_log_label': "ਨਵਾਂ ਲੌਗ ਐਂਟਰੀ", 'save_log_button': "ਲੌਗ ਸੇਵ ਕਰੋ", 'log_saved_success': "ਲੌਗ ਐਂਟਰੀ ਸਫਲਤਾਪੂਰਵਕ ਸੇਵ ਹੋਈ!", 'past_logs_header': "📖 ਪਿਛਲੀਆਂ ਐਂਟਰੀਆਂ", 'no_logs_message': "ਤੁਹਾਡੇ ਕੋਲ ਅਜੇ ਕੋਈ ਲੌਗ ਐਂਟਰੀਆਂ ਨਹੀਂ ਹਨ।", 'delete_log_button': "ਮਿਟਾਓ", 'log_deleted_success': "ਲੌਗ ਮਿਟਾ ਦਿੱਤਾ ਗਿਆ!", 'record_log_prompt': "ਇੱਕ ਵੌਇਸ ਨੋਟ ਜੋੜੋ:", 'voice_note_recorded': "🎤 ਵੌਇਸ ਨੋਟ ਰਿਕਾਰਡ ਕੀਤਾ ਗਿਆ ਹੈ ਅਤੇ ਤੁਹਾਡੇ ਲੌਗ ਨਾਲ ਸੇਵ ਕਰਨ ਲਈ ਤਿਆਰ ਹੈ।", 'attached_voice_note': "ਜੋੜਿਆ ਗਿਆ ਵੌਇਸ ਨੋਟ:",
        'upload_media_label': "ਤਸਵੀਰਾਂ ਜਾਂ ਵੀਡੀਓ ਜੋੜੋ:", 'attached_images': "ਜੋੜੀਆਂ ਗਈਆਂ ਤਸਵੀਰਾਂ:", 'attached_videos': "ਜੋੜੇ ਗਏ ਵੀਡੀਓ:",
        'forum_header': "👥 ਕਮਿਊਨਿਟੀ ਫੋਰਮ", 'forum_intro': "ਹੋਰ ਕਿਸਾਨਾਂ ਨਾਲ ਜੁੜੋ। ਸਵਾਲ ਪੁੱਛੋ, ਹੱਲ ਸਾਂਝੇ ਕਰੋ ਅਤੇ ਇਕੱਠੇ ਵਧੋ।",
        'create_post_header': "ਇੱਕ ਨਵੀਂ ਪੋਸਟ ਬਣਾਓ", 'post_title_label': "ਸਿਰਲੇਖ", 'post_title_placeholder': "ਇੱਕ ਸਪਸ਼ਟ ਅਤੇ ਸੰਖੇਪ ਸਿਰਲੇਖ ਦਾਖਲ ਕਰੋ", 'post_content_label': "ਤੁਹਾਡਾ ਸਵਾਲ ਜਾਂ ਸਲਾਹ", 'post_content_placeholder': "ਆਪਣੀ ਸਮੱਸਿਆ ਵਿਸਤਾਰਪੂਰਵਕ ਵਰਣਨ ਕਰੋ ਜਾਂ ਆਪਣਾ ਗਿਆਨ ਸਾਂਝਾ ਕਰੋ...",
        'upload_image_label': "ਇੱਕ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ (ਵਿਕਲਪਿਕ)", 'submit_post_button': "ਪੋਸਟ ਜਮ੍ਹਾਂ ਕਰੋ", 'post_success_message': "✅ ਤੁਹਾਡੀ ਪੋਸਟ ਪ੍ਰਕਾਸ਼ਿਤ ਹੋ ਗਈ ਹੈ!",
        'all_posts_header': "ਸਾਰੀਆਂ ਪੋਸਟਾਂ", 'posted_by': "{username} ਦੁਆਰਾ {date} ਨੂੰ ਪੋਸਟ ਕੀਤੀ ਗਈ", 'like_button': "👍 ਪਸੰਦ ({count})", 'reply_button': "💬 ਜਵਾਬ", 'replies_header': "ਜਵਾਬ", 'add_reply_placeholder': "ਇੱਕ ਜਵਾਬ ਲਿਖੋ...", 'submit_reply_button': "ਜਵਾਬ ਪੋਸਟ ਕਰੋ",
        'no_posts_message': "ਅਜੇ ਕੋਈ ਪੋਸਟਾਂ ਨਹੀਂ ਹਨ। ਗੱਲਬਾਤ ਸ਼ੁਰੂ ਕਰਨ ਵਾਲਾ ਪਹਿਲਾ ਵਿਅਕਤੀ ਬਣੋ!", 'view_replies': "{count} ਜਵਾਬ ਦੇਖੋ", 'hide_replies': "ਜਵਾਬ ਲੁਕਾਓ",
        'soil_characteristics_heading': "🔬 ਗੁਣ", 'soil_suitable_crops_heading': "🌱 ਢੁਕਵੀਆਂ ਫਸਲਾਂ", 'soil_model_not_found_warning': "⚠ ਮਿੱਟੀ ਭਵਿੱਖਬਾਣੀ ਮਾਡਲ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਪ੍ਰਸ਼ਾਸਕ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
        'stage_seedling': 'ਅੰਕੁਰਿਤ', 'stage_vegetative': 'ਉਦਭਿਦ', 'stage_flowering': 'ਫੁੱਲਣਾ', 'stage_fruiting': 'ਫਲਣਾ', 'stage_maturity': 'ਪਰਿਪੱਕਤਾ',
        'column_days_range': 'ਦਿਨਾਂ ਦੀ ਰੇਂਜ', 'column_fertilizer': 'ਖਾਦ', 'column_dosage': 'ਡੋਜ (ਕਿ.ਗ੍ਰਾ/ਏਕੜ)', 'column_frequency': 'ਫ੍ਰੀਕੁਐਂਸੀ', 'column_irrigation_depth': 'ਸਿੰਚਾਈ ਡੂੰਘਾਈ / ਮਾਤਰਾ', 'column_irrigation_frequency': 'ਸਿੰਚਾਈ ਫ੍ਰੀਕੁਐਂਸੀ',
    },
    'mr': {
        'app_title': "कृषीसखी", 'go_to': "जा", 'go_button': "जा", 'back_to_home': "घरी परत", 'language_selector_label': "भाषा",
        'page_home': "घर", 'page_crop_recommendation': "पिक शिफारस", 'page_yield_prediction': "उत्पादन अंदाज", 'page_crop_calendar': "पिक कॅलेंडर", 'page_weather_report': "हवामान अहवाल", 'page_leaf_disease_prediction': "पान रोग अंदाज", 'page_translator': "भाषांतरकार", 'page_profile': "प्रोफाइल", 'page_logs': "लॉग", 'page_community_forum': "समुदाय मंच",
        'quick_actions_title': "🚀 जलद क्रिया", 'quick_action_crop_desc': "आपल्या माती आणि हवामानावर आधारित योग्य पिक शिफारसी मिळवा।", 'quick_action_weather_desc': "आपल्या परिसराचे हवामान तपासा आणि आपल्या कृषी क्रियाकलापांची योजना करा।",
        'page_crop_guide': "पिक मार्गदर्शक", 'page_pest_prediction': "कीड अंदाज", 'page_soil_type_prediction': "माती प्रकार अंदाज",
        'crop_guide_title': "पिक खत आणि पाणी मार्गदर्शक", 'select_stage_label': "एक वाढ टप्पा निवडा", 'guide_table_header': "मार्गदर्शन तपशील",
        'file_not_found_error': "फाइल सापडली नाही. कृपया हे सुनिश्चित करा की ती योग्य फोल्डरमध्ये आहे।",
        'home_title': "कृषीसखीमध्ये आपले स्वागत! 🌱", 'home_subtitle': "आधुनिक शेतीसाठी आपला AI-चालित सहाय्यक।",
        'home_intro': "*कृषीसखी* शेतकऱ्यांना डेटा-चालित अंतर्दृष्टी प्रदान करण्यासाठी डिझाइन केले आहे.\nविविध वैशिष्ट्यांमध्ये प्रवेश करण्यासाठी साइडबारमधील विभागांद्वारे नेव्हिगेट करा:\n- *🌾 पिक शिफारस:* माती आणि पर्यावरणीय घटकांवर आधारित पेरण्यासाठी सर्वोत्तम पिकांची शिफारसी मिळवा.\n- *📈 उत्पादन अंदाज:* ऐतिहासिक डेटा आणि कृषी पॅरामीटर्सवर आधारित आपल्या पिकाच्या उत्पादनाचा अंदाज लावा.\n- *🗓 पिक कॅलेंडर:* केरळमधील विविध पिकांसाठी आदर्श पेरणी आणि कापणी वेळ पाहा.\n- *🌦 हवामान अहवाल:* आपल्या स्थानासाठी रिअल-टाइम हवामान अपडेट आणि अंदाज मिळवा.\n- *🌿 पान रोग अंदाज:* रोग लवकर ओळखण्यासाठी पानाचे प्रतिमा अपलोड करा.\n- *🌐 भाषांतरकार:* विविध भारतीय भाषांमध्ये कृषी शब्द आणि वाक्यांचे भाषांतर करा.\n- *👥 समुदाय मंच:* इतर शेतकऱ्यांशी जोडा, प्रश्न विचारा आणि उपाय सामायिक करा।",
        'crop_rec_header': "🌾 पिक शिफारस", 'crop_rec_intro': "पिक शिफारस मिळविण्यासाठी खालील तपशील प्रविष्ट करा।",
        'nitrogen_label': "नायट्रोजन (N)", 'phosphorus_label': "फॉस्फरस (P)", 'potassium_label': "पोटॅशियम (K)", 'temperature_label': "तापमान (°C)", 'humidity_label': "आर्द्रता (%)", 'ph_label': "pH मूल्य", 'rainfall_label': "पाऊस (mm)",
        'recommend_crop_button': "पिक शिफारस करा", 'recommendation_success': "🎉 शिफारस केलेले पिक *{crop}* आहे।",
        'crop_info_header': "{crop} बद्दल अधिक माहिती", 'description_heading': "📖 वर्णन", 'irrigation_heading': "💧 पाणी टिप्स", 'conditions_heading': "☀️ आदर्श परिस्थिती", 'pests_heading': "🐛 सामान्य कीड आणि रोग",
        'yield_pred_header': "📈 पिक उत्पादन अंदाज", 'yield_pred_intro': "केरळमधील विशिष्ट परिसरासाठी पिक उत्पादनाचा अंदाज लावण्यासाठी खालील तपशील भरा।",
        'year_label': "वर्ष", 'crop_label': "पिक", 'season_label': "हंगाम", 'area_label': "परिसर (हेक्टरमध्ये)",
        'predict_yield_button': "उत्पादन अंदाज लावा", 'yield_prediction_success': "🌾 अंदाज केलेले पिक उत्पादन सुमारे *{yield_val:.2f} किलो/हेक्टर* आहे।",
        'market_price_header': "💰 अंदाजे बाजार मूल्य", 'fetching_prices_spinner': "ताजी बाजार किंमती आणत आहे...", 'market_price_label': "सरासरी बाजार किंमत (प्रति क्विंटल)", 'total_value_label': "एकूण अंदाजे मूल्य", 'profit_header': "💵 अंदाजे नफा", 'cost_per_ha_label': "कृषी खर्च प्रति हेक्टर (₹)", 'total_cost_label': "एकूण कृषी खर्च", 'total_profit_label': "एकूण अंदाजे नफा", 'price_not_available': "{crop} साठी किंमत डेटा सध्या उपलब्ध नाही।", 'price_api_error': "सध्या बाजार किंमत डेटा मिळवू शकत नाही।",
        'crop_calendar_header': "🗓 पिक पेरणी आणि कापणी कॅलेंडर", 'select_crop_label': "एक पिक निवडा",
        'schedule_for_crop': "📅 {crop} साठी वेळापत्रक", 'sowing_start_label': "पेरणी प्रारंभ", 'sowing_end_label': "पेरणी समाप्ती", 'harvest_start_label': "कापणी प्रारंभ", 'harvest_end_label': "कापणी समाप्ती", 'fertiliser_label': "खत (N:P:K किलो/हेक्टर)", 'duration_label': "कालावधी",
        'weather_report_header': "🌦 हवामान अहवाल", 'enter_city_label': "आपल्या शहराचे नाव प्रविष्ट करा", 'get_weather_button': "हवामान मिळवा",
        'current_weather_in': "{location} मध्ये सध्याचे हवामान", 'temperature_metric': "तापमान", 'feels_like_metric': "वाटते", 'humidity_metric': "आर्द्रता", 'description_label': "वर्णन:", 'wind_label': "वारा:", 'precipitation_label': "पाऊस:",
        'forecast_header': "🗓 3-दिवसाचा अंदाज", 'avg_temp_label': "सरासरी तापमान:", 'max_temp_label': "कमाल तापमान:", 'min_temp_label': "किमान तापमान:", 'sunrise_label': "सूर्योदय:", 'sunset_label': "सूर्यास्त:", 'outlook_label': "दृष्टीकोन:",
        'error_parsing_weather': "हवामान डेटा पार्स करू शकत नाही. कृपया शहराचे नाव तपासा आणि पुन्हा प्रयत्न करा. त्रुटी: {e}", 'enter_city_warning': "⚠ कृपया शहराचे नाव प्रविष्ट करा।",
        'weather_fetch_error': "हवामान डेटा मिळवू शकत नाही. कृपया आपले नेटवर्क कनेक्शन किंवा शहराचे नाव तपासा आणि पुन्हा प्रयत्न करा।", 'weather_api_key_error': "API की त्रुटी: {message}. कृपया secrets.toml मध्ये आपली OpenWeatherMap API की तपासा. नवीन की सक्रिय होण्यास काही तास लागू शकतात।", 'weather_city_not_found_error': "शहर सापडले नाही त्रुटी: {message}. कृपया शहराच्या नावाची स्पेलिंग तपासा।", 'weather_api_generic_error': "API त्रुटी ({cod}): {message}",
        'translator_header': "🌐 शेतकरी भाषा भाषांतरकार", 'translator_intro': "✍ भाषांतर करण्यासाठी मजकूर प्रविष्ट करा", 'translate_button': "🔄 भाषांतर करा", 'translation_failed_error': "❌ भाषांतर अयशस्वी: {e}", 'enter_text_warning': "⚠ कृपया काही मजकूर प्रविष्ट करा।",
        'ai_chatbot_header': "🤖 AI चॅटबॉट", 'ai_chatbot_intro': "ही वैशिष्ट्य लवकरच येत आहे! आपल्या कृषी प्रश्नांमध्ये मदत करण्यासाठी एक शक्तिशाली AI चॅटबॉटची वाट पाहा।",
        'languages_supported_label': "समर्थित भाषा", 'farmers_helped_label': "मदत केलेले शेतकरी", 'crops_covered_label': "आच्छादित पिके", 'recommendations_given_label': "AI शिफारसी",
        'leaf_disease_header': "🌿 पान रोग अंदाज", 'leaf_disease_intro': "रोग तपासण्यासाठी झाडाच्या पानाचे प्रतिमा अपलोड करा।", 'model_not_found_warning': "⚠ रोग अंदाज मॉडेल सापडले नाही. कृपया प्रशासकाशी संपर्क साधा।",
        'uploader_label': "एक पान प्रतिमा निवडा...", 'uploaded_image_caption': "अपलोड केलेली प्रतिमा।", 'predict_disease_button': "🔍 रोग अंदाज लावा",
        'predicting_spinner': "पान विश्लेषण करत आहे...", 'prediction_result': "✅ अंदाज: *{disease}*",
        'remedy_info_header': "{disease} साठी उपचार आणि व्यवस्थापन", 'disease_description_heading': "ℹ️ रोगाबद्दल", 'organic_remedies_heading': "🌿 जैविक उपाय", 'chemical_remedies_heading': "🧪 रासायनिक उपाय", 'healthy_plant_message': "आनंदाची बातमी! आपले झाड निरोगी दिसत आहे. चांगल्या कृषी पद्धती कायम ठेवा।",
        'soil_prediction_header': "🏞️ माती प्रकार अंदाज", 'soil_prediction_intro': "मातीचा प्रकार ओळखण्यासाठी आणि त्याच्या गुणधर्मांबद्दल जाणून घेण्यासाठी मातीचे प्रतिमा अपलोड करा।", 'upload_soil_image_label': "एक माती प्रतिमा निवडा...",
        'predict_soil_button': "🔍 माती प्रकार अंदाज लावा", 'analyzing_soil_spinner': "माती विश्लेषण करत आहे...", 'soil_prediction_result': "✅ अंदाज केलेला माती प्रकार: *{soil_type}*", 'soil_info_header': "{soil_type} बद्दल माहिती",
        'pest_prediction_header': "प्रतिमेतून कीड ओळख", 'upload_image_pest': "कीड किंवा प्रभावित पानाचे एक प्रतिमा अपलोड करा", 'prediction_pest': "अंदाज केलेला कीड / कीटक", 'model_loading_error': "कीड अंदाज मॉडेल लोड करू शकत नाही. कृपया इंटरनेट कनेक्शन तपासा।",
        'remedy_info_header_pest': "{pest} साठी उपचार आणि व्यवस्थापन", 'pest_description_heading': "ℹ️ कीडाबद्दल", 'no_remedy_info': "{pest} साठी उपाय माहिती अद्याप उपलब्ध नाही।",
        'feedback_header': "अभिप्राय", 'rating_label': "आपल्या अनुभवाचे मूल्यांकन करा:", 'submit_button': "सबमिट करा", 'feedback_success': "आपल्या अभिप्रायाबद्दल धन्यवाद!",
        'contact_header': "संपर्क करा",
        'login_page_title': "कृषीसखीमध्ये स्वागत 🌱", 'login_tab': "लॉगिन", 'signup_tab': "साइन अप",
        'username_label': "वापरकर्तानाव", 'password_label': "पासवर्ड", 'login_button': "लॉगिन", 'signup_button': "साइन अप", 'logout_button': "लॉग आउट",
        'login_error': "चुकीचे वापरकर्तानाव किंवा पासवर्ड।", 'signup_success': "नोंदणी यशस्वी! कृपया लॉगिन करा।", 'signup_error_exists': "वापरकर्तानाव आधीपासून अस्तित्वात आहे. कृपया दुसरे निवडा।", 'signup_error_general': "नोंदणी दरम्यान एक त्रुटी आली।",
        'welcome_back': "परत स्वागत, {username}!",
        'profile_header': "👤 माझी प्रोफाइल", 'profile_subheader': "आपल्या खात्याचे तपशील आणि सेटिंग्ज व्यवस्थापित करा।",
        'full_name_label': "पूर्ण नाव", 'contact_number_label': "संपर्क क्रमांक", 'new_username_label': "नवीन वापरकर्तानाव", 'new_password_label': "नवीन पासवर्ड", 'confirm_password_label': "नवीन पासवर्डची पुष्टी करा", 'profile_pic_label': "प्रोफाइल प्रतिमा",
        'update_profile_button': "प्रोफाइल अपडेट करा", 'profile_update_success': "✅ प्रोफाइल यशस्वीरित्या अपडेट झाली!", 'profile_update_error': "❌ प्रोफाइल अपडेट करू शकत नाही. वापरकर्तानाव आधीपासून अस्तित्वात असू शकते।", 'password_mismatch_error': "पासवर्ड जुळत नाहीत।",
        'danger_zone_header': "🚨 धोकादायक क्षेत्र", 'delete_account_button': "माझे खाते हटवा", 'delete_account_warning': "ही क्रिया अपरिवर्तनीय आहे. आपला सर्व डेटा कायमचा हटवला जाईल।", 'delete_confirmation_label': "पुष्टी करण्यासाठी, कृपया खालील बॉक्समध्ये `DELETE` टाइप करा।",
        'delete_account_confirm_button': "होय, मी माझे खाते हटवू इच्छितो", 'account_deleted_success': "आपले खाते यशस्वीरित्या हटवले गेले आहे।", 'incorrect_delete_confirmation': "चुकीची पुष्टीकरण मजकूर।",
        'logs_header': "माझ्या कृषी लॉग", 'logs_intro': "येथे आपल्या दैनंदिन क्रियाकलाप, निरीक्षणे आणि नोट्स रेकॉर्ड करा।", 'new_log_label': "नवीन लॉग एंट्री", 'save_log_button': "लॉग सेव्ह करा", 'log_saved_success': "लॉग एंट्री यशस्वीरित्या सेव्ह झाली!", 'past_logs_header': "📖 मागील एंट्री", 'no_logs_message': "आपल्याकडे अद्याप कोणतीही लॉग एंट्री नाहीत।", 'delete_log_button': "हटवा", 'log_deleted_success': "लॉग हटवला!", 'record_log_prompt': "एक व्हॉइस नोट जोडा:", 'voice_note_recorded': "🎤 व्हॉइस नोट रेकॉर्ड केला गेला आहे आणि आपल्या लॉगसह सेव्ह करण्यासाठी तयार आहे।", 'attached_voice_note': "जोडलेला व्हॉइस नोट:",
        'upload_media_label': "फोटो किंवा व्हिडिओ जोडा:", 'attached_images': "जोडलेल्या प्रतिमा:", 'attached_videos': "जोडलेले व्हिडिओ:",
        'forum_header': "👥 समुदाय मंच", 'forum_intro': "इतर शेतकऱ्यांशी जोडा. प्रश्न विचारा, उपाय सामायिक करा आणि एकत्र वाढा।",
        'create_post_header': "एक नवीन पोस्ट तयार करा", 'post_title_label': "शीर्षक", 'post_title_placeholder': "एक स्पष्ट आणि संक्षिप्त शीर्षक प्रविष्ट करा", 'post_content_label': "आपला प्रश्न किंवा टिप", 'post_content_placeholder': "आपली समस्या तपशीलवार वर्णन करा किंवा आपले ज्ञान सामायिक करा...",
        'upload_image_label': "एक प्रतिमा अपलोड करा (पर्यायी)", 'submit_post_button': "पोस्ट सबमिट करा", 'post_success_message': "✅ आपली पोस्ट प्रकाशित झाली आहे!",
        'all_posts_header': "सर्व पोस्ट", 'posted_by': "{username} द्वारे {date} रोजी पोस्ट केले", 'like_button': "👍 आवड ({count})", 'reply_button': "💬 उत्तर", 'replies_header': "उत्तरे", 'add_reply_placeholder': "एक उत्तर लिहा...", 'submit_reply_button': "उत्तर पोस्ट करा",
        'no_posts_message': "अद्याप कोणत्याही पोस्ट नाहीत. संभाषण सुरू करणारा पहिला व्यक्ती व्हा!", 'view_replies': "{count} उत्तरे पाहा", 'hide_replies': "उत्तरे लपवा",
        'soil_characteristics_heading': "🔬 वैशिष्ट्ये", 'soil_suitable_crops_heading': "🌱 योग्य पिके", 'soil_model_not_found_warning': "⚠ माती अंदाज मॉडेल सापडले नाही. कृपया प्रशासकाशी संपर्क साधा।",
        'stage_seedling': 'रोप', 'stage_vegetative': 'वनस्पती', 'stage_flowering': 'फुलणं', 'stage_fruiting': 'फळधारणा', 'stage_maturity': 'परिपक्वता',
        'column_days_range': 'दिवसांची श्रेणी', 'column_fertilizer': 'खत', 'column_dosage': 'डोस (किलो/एकर)', 'column_frequency': 'वारंवारता', 'column_irrigation_depth': 'सिंचन खोली / प्रमाण', 'column_irrigation_frequency': 'सिंचन वारंवारता',
    },
}

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी',
    'ml': 'മലയാളം',
    'ta': 'தமிழ்',
    'te': 'తెలుగు',
    'bn': 'বাংলা',
    'mr': 'मराठी',
    'pa': 'ਪੰਜਾਬੀ',
}

TRANSLATOR_LANGUAGE_CODES = {
    'en': 'en',
    'hi': 'hi',
    'ml': 'ml',
    'ta': 'ta',
    'te': 'te',
    'bn': 'bn',
    'mr': 'mr',
    'pa': 'pa',
}

@st.cache_data(show_spinner=False)
def translate_text(text: str, target_lang: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    if target_lang == 'en':
        # If the target language is English, translate from auto-detected source to English.
        target_code = 'en'
    else:
        target_code = TRANSLATOR_LANGUAGE_CODES.get(target_lang, target_lang)
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception:
        return text


def translate_if_needed(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    lang = st.session_state.get('lang', 'en')
    if lang == 'en':
        return translate_text(text, 'en')
    return translate_text(text, lang)


def t(key, **kwargs):
    lang = st.session_state.get('lang', 'en')
    en_text = translations['en'].get(key, key)
    if lang == 'en':
        text = en_text
    else:
        text = translations.get(lang, {}).get(key, en_text)
        if text == en_text:
            text = translate_text(en_text, lang)
    return text.format(**kwargs) if kwargs else text

SOIL_INFO = {
    "alluvial soil": {
        "description": "Formed by the deposition of silt by rivers, these soils are highly fertile and found in river valleys and plains. They are rich in potash but often deficient in phosphorus and nitrogen.",
        "characteristics": "Light to dark in color, loamy texture, good porosity, and water retention. Very productive.",
        "suitable_crops": "Rice, wheat, sugarcane, cotton, jute, maize, oilseeds, and a variety of vegetables and fruits."
    },
    "black soil": {
        "description": "Also known as Regur soil or Black Cotton Soil, it's formed from the weathering of volcanic rocks. Ideal for cotton cultivation.",
        "characteristics": "High clay content, rich in iron, lime, and magnesium. Excellent water retention but poor drainage. Cracks deeply when dry.",
        "suitable_crops": "Cotton, sugarcane, jowar, tobacco, wheat, sunflower, and millets."
    },
    "clay soil": {
        "description": "Composed of very fine mineral particles and not much organic material. The particles bind together tightly, leading to poor drainage.",
        "characteristics": "Heavy, sticky when wet, hard when dry. High nutrient content but can be difficult to work with. Warms up slowly.",
        "suitable_crops": "Broccoli, cabbage, cauliflower, kale, peas, potatoes, and fruit trees that tolerate 'wet feet'."
    },
    "red soil": {
        "description": "Develops on crystalline igneous rocks in areas of low rainfall. The red color is due to the high concentration of iron oxide.",
        "characteristics": "Sandy to clayey texture, good drainage, but often poor in nutrients like nitrogen, phosphorus, and humus. Can be fertile with proper management.",
        "suitable_crops": "Wheat, cotton, pulses, tobacco, oilseeds, potatoes, millets, and groundnuts with proper irrigation and fertilization."
    },
    "laterite soil": {
        "description": "Found in areas with high temperature and heavy rainfall, leading to intense leaching of soil. Rich in iron and aluminum oxides.",
        "characteristics": "Acidic in nature, poor in nutrients, and has low water retention. Hardens like a brick when exposed to air.",
        "suitable_crops": "Tea, coffee, rubber, coconut, cashews, and areca nut. Not suitable for most other crops without heavy manuring."
    },
    "desert soil": {
        "description": "Also known as Arid soil, found in regions with low rainfall and high temperatures. Contains a high percentage of soluble salts.",
        "characteristics": "Sandy and gravelly texture, low in moisture and humus, alkaline in nature. Poor in organic matter and nitrogen.",
        "suitable_crops": "Drought-resistant crops like barley, bajra, guar, and some pulses can be grown with irrigation."
    },
    "peaty soil": {
        "description": "Formed in humid regions where organic matter accumulates. These soils are found in areas of heavy rainfall and high humidity.",
        "characteristics": "Dark, almost black color, high organic matter content, acidic. Can be very fertile but often suffers from poor drainage.",
        "suitable_crops": "Rice, jute, and sugarcane. Can be used for vegetables after treating the acidity and improving drainage."
    }
}

# =====================================================================================
# CROP AND DISEASE INFORMATION DATA
# =====================================================================================
CROP_INFO = {
    'rice': {
        'description': "A staple food for a large part of the world's human population, especially in Asia. It is the agricultural commodity with the third-highest worldwide production.",
        'irrigation': "Requires standing water for a significant period of its growth cycle. Fields should be flooded to a depth of 5-10 cm. Intermittent irrigation (wetting and drying) can save water without significantly impacting yield.",
        'conditions': "Prefers hot and humid climates. Ideal temperature range is 20°C to 37°C with high humidity and prolonged sunshine.",
        'pests': "Stem borer, leafhopper, brown planthopper, gall midge. Diseases include blast, sheath blight, and bacterial leaf blight."
    },
    'maize': {
        'description': "A cereal grain first domesticated by indigenous peoples in southern Mexico about 10,000 years ago. It has become a staple food in many parts of the world, with the total production of maize surpassing that of wheat or rice.",
        'irrigation': "Requires water at critical stages: tasseling, silking, and grain filling. Drip irrigation is highly efficient. Avoid water stress during the reproductive phase.",
        'conditions': "Grows best in warm climates. Ideal temperature is between 21°C and 27°C. Requires deep, fertile, and well-drained soil.",
        'pests': "Fall armyworm, corn earworm, stem borer. Diseases include downy mildew, leaf blight, and rust."
    },
    'jute': {
        'description': "A long, soft, shiny bast fiber that can be spun into coarse, strong threads. It is produced from flowering plants in the genus Corchorus.",
        'irrigation': "Requires significant rainfall and humidity. Supplemental irrigation is needed during dry spells, especially in the early stages of growth.",
        'conditions': "Grows well in alluvial soil and requires a warm, wet climate. Temperature range of 24°C to 37°C is ideal.",
        'pests': "Jute semilooper, yellow mite, stem weevil. Diseases include stem rot, anthracnose, and root rot."
    },
    'cotton': {
        'description': "A soft, fluffy staple fiber that grows in a boll, or protective case, around the seeds of the cotton plants of the genus Gossypium.",
        'irrigation': "Water requirement is high. Critical stages for irrigation are flowering and boll formation. Drip irrigation can improve water use efficiency and yield.",
        'conditions': "Requires a long frost-free period, plenty of sunshine, and moderate rainfall. Ideal temperatures are between 21°C and 37°C.",
        'pests': "Bollworm complex (American, pink, spotted), aphids, jassids, whitefly. Diseases include bacterial blight, fusarium wilt, and leaf curl virus."
    },
    'coconut': {
        'description': "A member of the palm tree family (Arecaceae) and the only known living species of the genus Cocos. The term 'coconut' can refer to the whole coconut palm, the seed, or the fruit.",
        'irrigation': "Requires regular watering, especially for young palms and during dry seasons. A well-irrigated palm can yield up to 30% more. Basin irrigation is a common method.",
        'conditions': "A tropical plant that thrives in sandy soils with high humidity and regular rainfall. Cannot tolerate cold temperatures.",
        'pests': "Rhinoceros beetle, red palm weevil, eriophyid mite. Diseases include bud rot, leaf rot, and Tatipaka disease."
    },
    'papaya': {
        'description': "The fruit of the plant Carica papaya. It is native to the tropics of the Americas, perhaps from southern Mexico and neighboring Central America.",
        'irrigation': "Needs regular watering but is very sensitive to waterlogging. Ensure good drainage. Drip irrigation is recommended. Water weekly in dry seasons.",
        'conditions': "Thrives in sunny, warm climates. Cannot tolerate frost. Well-drained soil is essential to prevent root rot.",
        'pests': "Aphids, mealybugs, spider mites. Diseases include papaya ringspot virus, anthracnose, and powdery mildew."
    },
    'orange': {
        'description': "The fruit of various citrus species in the family Rutaceae; it primarily refers to Citrus × sinensis, which is also called sweet orange.",
        'irrigation': "Young trees need frequent watering. Mature trees need deep watering at longer intervals. Water needs are highest during fruit set and development.",
        'conditions': "Prefers subtropical climates with moderate rainfall and sunshine. Well-drained, loamy soils are ideal.",
        'pests': "Citrus psylla, leaf miner, scale insects. Diseases include citrus canker, gummosis, and powdery mildew."
    },
    'apple': {
        'description': "A pome fruit produced by an apple tree (Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus.",
        'irrigation': "Requires consistent moisture, especially during the growing season. Drip irrigation is effective. Water stress can lead to smaller fruits.",
        'conditions': "Best suited for temperate climates with a distinct winter chilling period. Well-drained loamy soil is preferred.",
        'pests': "Codling moth, aphids, apple maggot. Diseases include apple scab, fire blight, and powdery mildew."
    },
    'muskmelon': {
        'description': "A species of melon that has been developed into many cultivated varieties. These varieties include smooth-skinned varieties like honeydew, crenshaw, and casaba, and netted-skin varieties.",
        'irrigation': "Needs consistent watering, especially during fruit development. Avoid wetting the leaves and fruit to prevent diseases. Drip irrigation is ideal.",
        'conditions': "A warm-season crop that requires a long growing season of hot, sunny weather. Sandy loam soils are best.",
        'pests': "Aphids, cucumber beetles, squash bugs. Diseases include powdery mildew, downy mildew, and fusarium wilt."
    },
    'watermelon': {
        'description': "A flowering plant species of the Cucurbitaceae family and the name of its edible fruit. A scrambling and trailing vine-like plant, it is a highly cultivated fruit worldwide, having more than 1,000 varieties.",
        'irrigation': "Requires consistent and plentiful water, especially during the period from planting to when fruits begin to form. Reduce watering as fruits ripen to increase sugar content.",
        'conditions': "Needs hot, sunny conditions to thrive. Prefers well-drained, sandy loam soil. Long growing season is a must.",
        'pests': "Aphids, cucumber beetles, vine borers. Diseases include anthracnose, powdery mildew, and downy mildew."
    },
    'grapes': {
        'description': "A fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis.",
        'irrigation': "Deep watering at intervals is better than frequent shallow watering. Drip irrigation is highly recommended to keep foliage dry and prevent fungal diseases.",
        'conditions': "Require a long season of warm to hot, dry weather and full sun. Well-drained soil is crucial.",
        'pests': "Grapevine moth, mealybugs, spider mites. Diseases include powdery mildew, downy mildew, and black rot."
    },
    'mango': {
        'description': "An edible stone fruit produced by the tropical tree Mangifera indica. It is believed to have originated in the region between northwestern Myanmar, Bangladesh, and northeastern India.",
        'irrigation': "Irrigation is not required during the monsoon. However, regular watering from fruit set to maturity is crucial for good yield and quality. Stop watering a few weeks before harvest.",
        'conditions': "A tropical fruit that prefers climates with a distinct dry season for good fruit production. Cannot tolerate frost.",
        'pests': "Mango hopper, mealybug, fruit fly. Diseases include anthracnose, powdery mildew, and malformation."
    },
    'banana': {
        'description': "An elongated, edible fruit – botanically a berry – produced by several kinds of large herbaceous flowering plants in the genus Musa.",
        'irrigation': "A water-intensive crop. Requires frequent irrigation to maintain soil moisture, especially during the vegetative growth phase. Drip irrigation is very effective.",
        'conditions': "Thrives in tropical and subtropical regions. Requires rich, well-drained soil and high humidity.",
        'pests': "Rhizome weevil, banana aphid, stem borer. Diseases include Panama wilt, Sigatoka leaf spot, and bunchy top virus."
    },
    'pomegranate': {
        'description': "A fruit-bearing deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows between 5 and 10 m tall.",
        'irrigation': "Drought-tolerant once established, but regular irrigation is necessary for good fruit production. Drip irrigation is the most suitable method.",
        'conditions': "Prefers a semi-arid climate. Can tolerate hot, dry summers and cold winters. Well-drained soil is essential.",
        'pests': "Pomegranate butterfly (fruit borer), aphids, mealybugs. Diseases include bacterial blight, leaf spot, and fruit rot."
    },
    'lentil': {
        'description': "An edible legume. It is an annual plant known for its lens-shaped seeds. It is about 40 cm tall, and the seeds grow in pods, usually with two seeds in each.",
        'irrigation': "Generally grown as a rainfed crop. One or two irrigations at flowering and pod-filling stages can significantly boost yield in dry conditions.",
        'conditions': "A cool-season crop. Prefers well-drained, loamy soils. Can tolerate drought to some extent.",
        'pests': "Aphids, pod borer. Diseases include rust, wilt, and blight."
    },
    'blackgram': {
        'description': "A bean grown in the Indian subcontinent. It, along with the mung bean, was placed in the genus Vigna, and is still often seen as V. mungo.",
        'irrigation': "Mostly grown as a rainfed crop. If necessary, one irrigation at the flowering stage is beneficial.",
        'conditions': "Warm climate crop. Grows best in the Kharif season. Requires well-drained loamy or sandy loam soil.",
        'pests': "Pod borer, whitefly. Diseases include yellow mosaic virus, powdery mildew, and leaf spot."
    },
    'mungbean': {
        'description': "A plant species in the legume family. The mung bean is mainly cultivated in East Asia, Southeast Asia and the Indian subcontinent.",
        'irrigation': "Short-duration crop, generally grown under rainfed conditions. One irrigation may be needed if there is a long dry spell during flowering.",
        'conditions': "Requires a warm climate. Best grown during summer and Kharif seasons. Prefers well-drained loamy soil.",
        'pests': "Pod borer, thrips. Diseases include yellow mosaic virus and powdery mildew."
    },
    'mothbeans': {
        'description': "A drought-resistant legume, commonly grown in arid and semi-arid regions of India. It is a creeping annual vine.",
        'irrigation': "Highly drought-resistant and typically grown without irrigation.",
        'conditions': "Thrives in hot, dry, sandy soils. Extremely well-adapted to arid conditions.",
        'pests': "Generally resistant to pests. Whitefly can be an issue. Diseases are rare."
    },
    'pigeonpeas': {
        'description': "A perennial legume from the family Fabaceae. Since its domestication in the Indian subcontinent at least 3,500 years ago, its seeds have become a common food in Asia, Africa, and Latin America.",
        'irrigation': "Deep-rooted and drought-resistant. Generally grown as a rainfed crop. One irrigation during pod development can increase yield.",
        'conditions': "A tropical and subtropical crop. Requires well-drained soil and cannot tolerate waterlogging.",
        'pests': "Pod borer, pod fly, maruca. Diseases include fusarium wilt and sterility mosaic disease."
    },
    'kidneybeans': {
        'description': "A variety of the common bean (Phaseolus vulgaris). It is named for its visual resemblance in shape and color to a kidney.",
        'irrigation': "Requires consistent moisture throughout the growing season. Water stress during flowering can severely reduce yield.",
        'conditions': "A warm-season crop. Prefers well-drained, light soils. Sensitive to both frost and high temperatures.",
        'pests': "Bean beetle, aphids. Diseases include anthracnose, bean common mosaic virus, and rust."
    },
    'chickpea': {
        'description': "An annual legume of the family Fabaceae, subfamily Faboideae. Its different types are variously known as gram or Bengal gram, garbanzo or garbanzo bean, or Egyptian pea.",
        'irrigation': "Typically grown as a rainfed crop in the Rabi season. One pre-sowing irrigation and one at the flowering stage are beneficial in areas with low winter rainfall.",
        'conditions': "A cool, dry climate crop. Cannot tolerate frost, especially at the flowering stage. Prefers light, well-drained soils.",
        'pests': "Pod borer (Helicoverpa armigera) is the major pest. Diseases include fusarium wilt, ascochyta blight, and rust."
    },
    'coffee': {
        'description': "A brewed drink prepared from roasted coffee beans, the seeds of berries from certain flowering plants in the Coffea genus.",
        'irrigation': "Irrigation is crucial, especially for triggering uniform flowering after a dry spell ('blossom showers'). Drip irrigation is widely adopted to conserve water.",
        'conditions': "Requires a specific altitude, temperature, and rainfall. Prefers rich, well-drained soil and is often grown under shade.",
        'pests': "Coffee berry borer, white stem borer. Diseases include coffee leaf rust and black rot."
    }
}

DISEASE_REMEDIES = {
    "apple scab": {
        "description": "A fungal disease that causes olive-green to brown spots on leaves, fruit, and twigs. Severe infections can cause defoliation and disfigured fruit.",
        "organic": "Prune and destroy infected twigs. Rake up and burn fallen leaves. Spray with sulfur, neem oil, or potassium bicarbonate solutions.",
        "chemical": "Apply fungicides containing Captan, Myclobutanil, or Mancozeb. Start applications at green tip and continue through petal fall."
    },
    "apple black rot": {
        "description": "A fungal disease causing 'frog-eye' leaf spots, limb cankers, and a firm, brown-to-black rot on the fruit, often starting at the blossom end.",
        "organic": "Prune out cankered branches. Remove mummified fruit from the tree and the ground. Improve air circulation through pruning.",
        "chemical": "Fungicides like Captan, Thiophanate-methyl, and Pristine are effective. Apply during the bloom period and as a cover spray."
    },
    "apple cedar rust": {
        "description": "A fungal disease that requires both apple and cedar/juniper trees to complete its life cycle. It creates bright orange-yellow spots on apple leaves and fruit.",
        "organic": "Remove nearby cedar and juniper trees if possible. Apply sulfur or neem oil sprays. Plant rust-resistant apple varieties.",
        "chemical": "Fungicides containing Myclobutanil or Ferbam are effective. Apply from pink bud stage until mid-summer."
    },
    "corn gray leaf spot": {
        "description": "A fungal disease characterized by long, narrow, rectangular, tan-colored lesions that run parallel to the leaf veins.",
        "organic": "Use resistant corn hybrids. Practice crop rotation with non-host crops. Tillage to bury infected crop residue.",
        "chemical": "Apply fungicides such as Pyraclostrobin, Azoxystrobin, or Propiconazole, especially if the disease is present on upper leaves during tasseling."
    },
    "corn common rust": {
        "description": "Caused by a fungus, it appears as small, cinnamon-brown, powdery pustules on both upper and lower leaf surfaces.",
        "organic": "Plant resistant hybrids. Early planting can sometimes help the crop mature before rust becomes severe.",
        "chemical": "Foliar fungicides (e.g., Propiconazole, Azoxystrobin) are effective but often not economically necessary unless the infection is severe on a susceptible hybrid."
    },
    "corn northern leaf blight": {
        "description": "A fungal disease that produces long, elliptical, grayish-green or tan lesions on the leaves, leading to significant yield loss.",
        "organic": "Plant resistant hybrids. Crop rotation and tillage to reduce residue are key management practices.",
        "chemical": "Fungicide applications with products containing Pyraclostrobin or Propiconazole can be effective if applied early."
    },
    "grape black rot": {
        "description": "A serious fungal disease of grapes. It attacks all green parts of the vine – leaves, shoots, tendrils, and fruit. The fruit turns black, hard, and mummified.",
        "organic": "Sanitation is crucial: remove and destroy all mummified fruit and infected canes. Improve air circulation. Apply copper or sulfur sprays.",
        "chemical": "Apply fungicides like Mancozeb, Captan, or Myclobutanil. Start sprays before bloom and continue until fruit begins to color."
    },
    "grape esca (black measles)": {
        "description": "A complex fungal disease that causes a slow decline of the vine. Symptoms include tiger-striped patterns on leaves and small, dark spots on berries.",
        "organic": "Prune out and destroy infected or dead wood. Double pruning (late initial pruning followed by a second pass) can sometimes help. No effective organic sprays.",
        "chemical": "Currently, there are no highly effective chemical treatments. Management relies on surgical removal of infected wood."
    },
    "grape leaf blight": {
        "description": "Also known as Isariopsis leaf spot, this fungal disease causes dark brown, irregular spots on leaves, which may develop a yellow halo.",
        "organic": "Improve air circulation through proper pruning and canopy management. Rake and destroy fallen leaves.",
        "chemical": "Fungicides used for black rot and downy mildew (e.g., Mancozeb) are typically effective against leaf blight as well."
    },
    "potato early blight": {
        "description": "A fungal disease that causes dark, target-like spots on lower, older leaves. Lesions can also appear on stems and tubers.",
        "organic": "Crop rotation, proper plant spacing for air circulation, and destroying infected crop debris. Copper-based fungicides can be used.",
        "chemical": "Apply fungicides such as Chlorothalonil, Mancozeb, or Azoxystrobin preventively, especially during warm, humid conditions."
    },
    "potato late blight": {
        "description": "A devastating water mold disease that causes large, dark, water-soaked lesions on leaves and stems. It can destroy an entire crop rapidly.",
        "organic": "Plant certified disease-free seed potatoes. Ensure good drainage. Destroy volunteer potato plants. Copper sprays have some effect if applied preventively.",
        "chemical": "Aggressive fungicide programs are essential. Use products like Chlorothalonil, Mancozeb, and specialized products like Ridomil (Mefenoxam) for systemic control."
    },
    "tomato bacterial spot": {
        "description": "A bacterial disease causing small, water-soaked, dark spots on leaves and fruit. The spots on leaves may have a yellow halo.",
        "organic": "Use disease-free seed and transplants. Rotate crops. Avoid overhead irrigation. Sprays with copper-based bactericides can help suppress the disease.",
        "chemical": "Preventive applications of copper mixed with Mancozeb. Some growers use products containing Acibenzolar-S-methyl to boost plant defenses."
    },
    "tomato early blight": {
        "description": "A fungal disease that appears as dark, target-like spots on lower leaves, often accompanied by yellowing. Can also cause stem lesions.",
        "organic": "Prune lower leaves. Mulch around the base of plants. Use drip irrigation. Copper and bio-fungicides (Bacillus subtilis) can be effective.",
        "chemical": "Fungicides containing Chlorothalonil, Mancozeb, or Azoxystrobin. Begin applications when conditions are favorable for disease."
    },
    "tomato late blight": {
        "description": "The same water mold that affects potatoes. It causes large, greasy, grey-green spots on leaves that quickly turn brown. A white fuzzy mold may appear on the underside.",
        "organic": "Ensure good air circulation. Avoid overhead watering. Copper sprays can provide some protection if applied before the disease appears.",
        "chemical": "Requires a strict fungicide schedule. Products include Chlorothalonil, Mancozeb, and specialized systemic fungicides."
    },
    "tomato leaf mold": {
        "description": "A fungal disease that primarily affects greenhouse tomatoes. It causes pale green or yellowish spots on the upper leaf surface and olive-green to brown velvety mold on the underside.",
        "organic": "Improve ventilation and reduce humidity. Prune lower leaves. Plant resistant varieties.",
        "chemical": "Fungicides like Chlorothalonil or those containing Strobilurins can be effective."
    },
    "tomato septoria leaf spot": {
        "description": "A fungal disease that causes numerous small, circular spots with dark borders and tan centers on the lower leaves. Small black dots (pycnidia) can be seen in the centers.",
        "organic": "Remove and destroy infected lower leaves. Mulch heavily. Use drip irrigation. Copper-based sprays can help.",
        "chemical": "Apply fungicides containing Chlorothalonil or Mancozeb."
    },
    "tomato spider mites": {
        "description": "Not a disease, but a pest. These tiny arachnids suck cell contents from leaves, causing a fine, yellowish stippling. Fine webbing may be visible on the underside of leaves.",
        "organic": "A strong jet of water can dislodge them. Release predatory mites. Apply insecticidal soap or neem oil.",
        "chemical": "Use miticides like Abamectin or Spiromesifen. Rotate chemical classes to prevent resistance."
    },
    "tomato target spot": {
        "description": "A fungal disease causing small, water-soaked spots on leaves that enlarge into target-like lesions with concentric rings.",
        "organic": "Improve air circulation. Mulch the soil. Avoid overhead irrigation. Copper fungicides can be used.",
        "chemical": "Fungicides such as Chlorothalonil, Mancozeb, or those in the Strobilurin class are effective."
    },
    "tomato yellow leaf curl virus": {
        "description": "A viral disease transmitted by whiteflies. It causes severe stunting, upward curling and yellowing of leaves, and reduced fruit set.",
        "organic": "Control whitefly populations using yellow sticky traps, neem oil, or insecticidal soap. Remove and destroy infected plants immediately.",
        "chemical": "Use systemic insecticides like Imidacloprid to control whiteflies. There is no cure for the virus itself."
    },
    "tomato mosaic virus": {
        "description": "A viral disease causing a mosaic of light green and yellow patterns on the leaves, along with some distortion and stunting.",
        "organic": "There is no cure. Practice good sanitation. Wash hands before handling plants. Remove and destroy infected plants. Do not use tobacco products near tomato plants.",
        "chemical": "No chemical cure. Management focuses on preventing transmission by controlling aphids and practicing good hygiene."
    }
}

PEST_REMEDIES = {
    "aphid": {
        "description": "Small, sap-sucking insects that cluster on new growth, causing leaves to curl and yellow. They excrete a sticky 'honeydew' that can lead to sooty mold.",
        "organic": "Spray with a strong jet of water to dislodge them. Introduce beneficial insects like ladybugs or lacewings. Apply neem oil or insecticidal soap.",
        "chemical": "Use insecticides containing Imidacloprid, Acetamiphrid, or Malathion. Follow label instructions carefully."
    },
    "caterpillar": {
        "description": "The larval stage of moths and butterflies. Many species are voracious eaters of leaves, stems, and fruits, causing significant crop damage.",
        "organic": "Handpick them off plants. Use Bacillus thuringiensis (Bt), a natural bacterium that is toxic to caterpillars. Apply neem oil or Spinosad.",
        "chemical": "Insecticides like Emamectin Benzoate, Chlorantraniliprole, or Lambda-cyhalothrin are effective. Target young larvae for best results."
    },
    "grasshopper": {
        "description": "Chewing insects that can consume large amounts of plant foliage. Large populations can quickly defoliate entire fields.",
        "organic": "Use floating row covers on young plants. Encourage natural predators like birds. Apply sprays containing Nosema locustae spores or garlic oil.",
        "chemical": "Use baits containing Carbaryl or apply insecticides like Bifenthrin or Deltamethrin, especially around field borders."
    },
    "whitefly": {
        "description": "Tiny, winged insects that suck plant juices and transmit plant viruses. They are typically found on the undersides of leaves.",
        "organic": "Use yellow sticky traps to monitor and trap adults. Apply neem oil or insecticidal soap, ensuring good coverage on the undersides of leaves. Introduce beneficial insects like Encarsia formosa.",
        "chemical": "Use insecticides such as Spiromesifen, Diafenthiuron, or Imidacloprid. Rotate chemical classes to manage resistance."
    },
    "thrip": {
        "description": "Minute, slender insects with fringed wings. They feed by puncturing plant cells and sucking out the contents, causing stippling, discoloration, and deformation of leaves and flowers.",
        "organic": "Use blue or yellow sticky traps. Release predatory mites (Amblyseius spp.). Apply neem oil or Spinosad sprays.",
        "chemical": "Insecticides like Fipronil or Acetamiphrid can be used. Systemic insecticides are often more effective."
    },
    "spider mite": {
        "description": "Tiny arachnids (not insects) that are difficult to see with the naked eye. They cause a fine, yellowish stippling on leaves and may produce fine webbing. Thrive in hot, dry conditions.",
        "organic": "Increase humidity by spraying plants with water. Release predatory mites like Phytoseiulus persimilis. Apply horticultural oil or insecticidal soap.",
        "chemical": "Use miticides containing Abamectin, Propargite, or Fenazaquin. Ensure thorough coverage, especially on the underside of leaves."
    },
    "leaf beetle": {
        "description": "A common pest that chews holes in the leaves of various plants, especially vegetables like beans, potatoes, and eggplant. Heavy infestations can lead to skeletonized leaves and reduced plant vigor.",
        "organic": "Handpick the beetles and drop them into soapy water. Use floating row covers to protect young plants. Apply neem oil or Spinosad, ensuring to cover the underside of leaves.",
        "chemical": "Insecticides containing Bifenthrin, Cypermethrin, or Carbaryl are effective. It's best to spray in the evening when bees are not active."
    },
    "aphid": {
        "description": "Small, sap-sucking insects that cluster on new growth, causing leaves to curl and yellow. They excrete a sticky 'honeydew' that can lead to sooty mold.",
        "organic": "Spray with a strong jet of water to dislodge them. Introduce beneficial insects like ladybugs or lacewings. Apply neem oil or insecticidal soap.",
        "chemical": "Use insecticides containing Imidacloprid, Acetamiphrid, or Malathion. Follow label instructions carefully."
    },
    "caterpillar": {
        "description": "The larval stage of moths and butterflies. Many species are voracious eaters of leaves, stems, and fruits, causing significant crop damage.",
        "organic": "Handpick them off plants. Use Bacillus thuringiensis (Bt), a natural bacterium that is toxic to caterpillars. Apply neem oil or Spinosad.",
        "chemical": "Insecticides like Emamectin Benzoate, Chlorantraniliprole, or Lambda-cyhalothrin are effective. Target young larvae for best results."
    },
    "grasshopper": {
        "description": "Chewing insects that can consume large amounts of plant foliage. Large populations can quickly defoliate entire fields.",
        "organic": "Use floating row covers on young plants. Encourage natural predators like birds. Apply sprays containing Nosema locustae spores or garlic oil.",
        "chemical": "Use baits containing Carbaryl or apply insecticides like Bifenthrin or Deltamethrin, especially around field borders."
    },
    "whitefly": {
        "description": "Tiny, winged insects that suck plant juices and transmit plant viruses. They are typically found on the undersides of leaves.",
        "organic": "Use yellow sticky traps to monitor and trap adults. Apply neem oil or insecticidal soap, ensuring good coverage on the undersides of leaves. Introduce beneficial insects like Encarsia formosa.",
        "chemical": "Use insecticides such as Spiromesifen, Diafenthruron, or Imidacloprid. Rotate chemical classes to manage resistance."
    },
    "thrip": {
        "description": "Minute, slender insects with fringed wings. They feed by puncturing plant cells and sucking out the contents, causing stippling, discoloration, and deformation of leaves and flowers.",
        "organic": "Use blue or yellow sticky traps. Release predatory mites (Amblyseius spp.). Apply neem oil or Spinosad sprays.",
        "chemical": "Insecticides like Fipronil or Acetamiphrid can be used. Systemic insecticides are often more effective."
    },
    "spider mite": {
        "description": "Tiny arachnids (not insects) that are difficult to see with the naked eye. They cause a fine, yellowish stippling on leaves and may produce fine webbing. Thrive in hot, dry conditions.",
        "organic": "Increase humidity by spraying plants with water. Release predatory mites like Phytoseiulus persimilis. Apply horticultural oil or insecticidal soap.",
        "chemical": "Use miticides containing Abamectin, Propargite, or Fenazaquin. Ensure thorough coverage, especially on the underside of leaves."
    }
}

# =====================================================================================
# 2. SESSION STATE INITIALIZATION
# =====================================================================================
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

# Keep the language selector widgets in sync with the currently selected language
if 'login_lang_selector' not in st.session_state:
    st.session_state['login_lang_selector'] = st.session_state.get('lang', 'en')
if 'lang_selector' not in st.session_state:
    st.session_state['lang_selector'] = st.session_state.get('lang', 'en')

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

def t(key, **kwargs):
    lang = st.session_state.get('lang', 'en')
    en_text = translations['en'].get(key, key)
    lang_dict = translations.get(lang, translations['en'])
    text = lang_dict.get(key, en_text)
    if text == en_text and lang != 'en':
        text = translate_text(en_text, lang)
    return text.format(**kwargs) if kwargs else text

# Floating home button (inserted after translator helper is defined so t() works)
st.markdown(f"""
""", unsafe_allow_html=True)


def set_login_language():
    st.session_state['lang'] = st.session_state['login_lang_selector']
    st.session_state['lang_selector'] = st.session_state['login_lang_selector']

# =====================================================================================
# 3. MONGODB AND AUTHENTICATION SETUP (Updated with Forum collections and functions)
# =====================================================================================
# Use pbkdf2_sha256 so long passwords are handled safely without bcrypt's 72-byte limit.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

@st.cache_resource
def init_connection():
    try:
        uri = os.getenv("mongodb_uri") or st.secrets.get("mongo_uri") or st.secrets.get("mongodb_uri")
        if not uri:
            raise ValueError("Missing MongoDB URI in environment variable 'mongodb_uri' or Streamlit secrets.")
        client = pymongo.MongoClient(uri)
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB. Please ensure your connection string is correctly configured in secrets.toml or environment variables. Error: {e}")
        return None

client = init_connection()

if client:
    db = client.krishisakhi_db
    users_collection = db.users
    user_logs_collection = db.user_logs
    forum_posts_collection = db.forum_posts
    forum_replies_collection = db.forum_replies

def send_sms(number, message, schedule_time=None):
    """Sends an SMS using the FAST2SMS API."""
    try:
        # Check if API key exists
        try:
            api_key = st.secrets["fast2sms_api_key"]
        except KeyError:
            st.error("❌ FAST2SMS API key not found in secrets.toml")
            st.info("📝 Add your API key to `.streamlit/secrets.toml` like this:")
            st.code("fast2sms_api_key = \"your_api_key_here\"", language="toml")
            return False

        # Validate API key format
        if not api_key or len(api_key.strip()) < 10:
            st.error("❌ FAST2SMS API key appears to be invalid (too short)")
            st.info("🔑 Get your API key from: https://www.fast2sms.com/")
            return False

        url = "https://www.fast2sms.com/dev/bulkV2"
        params = {
            "authorization": api_key,
            "route": "q",
            "message": message,
            "numbers": number,
            "flash": "0"
        }
        if schedule_time:
            params["schedule_time"] = schedule_time

        st.info(f"📡 Sending SMS to {number}...")
        response = requests.get(url, params=params, timeout=15)

        # Debug information
        st.write(f"**HTTP Status:** {response.status_code}")

        # Check response status
        if response.status_code == 401:
            st.error("❌ API Key Invalid (401 Unauthorized)")
            st.info("🔧 Solutions:")
            st.info("1. Check your API key in `.streamlit/secrets.toml`")
            st.info("2. Get a new API key from https://www.fast2sms.com/")
            st.info("3. Make sure your account has SMS credits")
            return False
        elif response.status_code == 400:
            st.error("❌ Bad Request (400)")
            st.info("🔧 Check phone number format and message content")
            return False
        elif response.status_code != 200:
            st.error(f"❌ API Error: HTTP {response.status_code}")
            if response.text:
                st.error(f"Response: {response.text[:300]}...")
            return False

        # Check if response is empty
        if not response.text.strip():
            st.error("❌ Empty response from FAST2SMS API")
            st.info("🔧 This usually means:")
            st.info("1. API key is expired")
            st.info("2. Account has no SMS credits")
            st.info("3. API service is temporarily down")
            return False

        # Try to parse JSON
        try:
            response_data = response.json()
        except ValueError as e:
            st.error(f"❌ Invalid JSON response: {str(e)}")
            st.error(f"Raw response: {response.text[:300]}...")
            st.info("🔧 The API returned HTML instead of JSON. Check your API key.")
            return False

        # Check API response
        if response_data.get("return") is True:
            return True
        else:
            error_message = response_data.get("message", "Unknown API error")
            st.error(f"❌ FAST2SMS API Error: {error_message}")
            # Show additional debugging info
            st.write("**Full API Response:**", response_data)
            return False

    except requests.exceptions.Timeout:
        st.error("❌ Request timeout - please try again")
        st.info("🔧 Check your internet connection")
        return False
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error - check internet connection")
        st.info("🔧 The API server might be down")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return False

def send_crop_reminder_sms(number, crop_name, crop_info):
    """Sends a comprehensive SMS reminder with sowing and harvesting details using FAST2SMS API."""
    try:
        # Check if API key exists
        try:
            api_key = st.secrets["fast2sms_api_key"]
        except KeyError:
            st.error("❌ FAST2SMS API key not found in secrets.toml")
            st.info("📝 Add your API key to `.streamlit/secrets.toml` like this:")
            st.code("fast2sms_api_key = \"your_api_key_here\"", language="toml")
            return False

        # Validate API key format
        if not api_key or len(api_key.strip()) < 10:
            st.error("❌ FAST2SMS API key appears to be invalid (too short)")
            st.info("🔑 Get your API key from: https://www.fast2sms.com/")
            return False

        url = "https://www.fast2sms.com/dev/bulkV2"

        # Comprehensive message with sowing and harvesting details
        message = (
            f"KrishiSakhi Reminder - {crop_name}\n\n"
            f"SOWING WINDOW:\n"
            f"Start: {crop_info['Sowing Start (Kerala)']}\n"
            f"End: {crop_info['Sowing End (Kerala)']}\n\n"
            f"CROP DURATION: {crop_info['Duration (Days)']} days\n\n"
            f"HARVEST WINDOW:\n"
            f"Start: {crop_info['Harvest Start (Approx)']}\n"
            f"End: {crop_info['Harvest End (Approx)']}\n\n"
            f"Fertiliser (N:P:K kg/ha): {crop_info['Fertiliser']}\n\n"
            f"Happy Farming!"
        )

        # Translate the SMS text into the selected app language if needed
        current_lang = st.session_state.get('lang', 'en')
        if current_lang != 'en':
            message = translate_text(message, current_lang)

        params = {
            "authorization": api_key,
            "route": "q",
            "message": message,
            "numbers": number,
            "flash": "0"
        }

        st.info(f"📡 Sending crop reminder SMS to {number}...")
        response = requests.get(url, params=params, timeout=15)

        # Debug information
        st.write(f"**HTTP Status:** {response.status_code}")

        # Check response status
        if response.status_code == 401:
            st.error("❌ API Key Invalid (401 Unauthorized)")
            st.info("🔧 Solutions:")
            st.info("1. Check your API key in `.streamlit/secrets.toml`")
            st.info("2. Get a new API key from https://www.fast2sms.com/")
            st.info("3. Make sure your account has SMS credits")
            return False
        elif response.status_code == 400:
            st.error("❌ Bad Request (400)")
            st.info("🔧 Check phone number format and message content")
            return False
        elif response.status_code != 200:
            st.error(f"❌ API Error: HTTP {response.status_code}")
            if response.text:
                st.error(f"Response: {response.text[:300]}...")
            return False

        # Check if response is empty
        if not response.text.strip():
            st.error("❌ Empty response from FAST2SMS API")
            st.info("🔧 This usually means:")
            st.info("1. API key is expired")
            st.info("2. Account has no SMS credits")
            st.info("3. API service is temporarily down")
            return False

        # Try to parse JSON
        try:
            response_data = response.json()
        except ValueError as e:
            st.error(f"❌ Invalid JSON response: {str(e)}")
            st.error(f"Raw response: {response.text[:300]}...")
            st.info("🔧 The API returned HTML instead of JSON. Check your API key.")
            return False

        # Check API response
        if response_data.get("return") is True:
            return True
        else:
            error_message = response_data.get("message", "Unknown API error")
            st.error(f"❌ FAST2SMS API Error: {error_message}")
            # Show additional debugging info
            st.write("**Full API Response:**", response_data)
            return False

    except requests.exceptions.Timeout:
        st.error("❌ Request timeout - please try again")
        st.info("🔧 Check your internet connection")
        return False
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error - check internet connection")
        st.info("🔧 The API server might be down")
def verify_password(plain_password: str, hashed_password: str):
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except UnknownHashError:
        if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        raise

def hash_password(password: str):
    return pwd_context.hash(password)

def user_exists(username: str):
    if client:
        return users_collection.find_one({"username": username}) is not None
    return False

def add_user(username: str, password: str, full_name: str):
    if client and not user_exists(username):
        hashed_pass = hash_password(password)
        users_collection.insert_one({
            "username": username,
            "password": hashed_pass,
            "full_name": full_name,
            "profile_pic_base64": "",
            "contact_number": "" 
        })
        return True
    return False

def get_user_details(username: str):
    if client:
        return users_collection.find_one({"username": username})
    return None

def update_user_profile(current_username, updates):
    if not client:
        return False
        
    if 'username' in updates and updates['username'] != current_username:
        if user_exists(updates['username']):
            return False 
    
    if 'password' in updates and updates['password']:
        updates['password'] = hash_password(updates['password'])
    else:
        updates.pop('password', None)
        
    users_collection.update_one({"username": current_username}, {"$set": updates})
    return True

def delete_user(username: str):
    if client:
        users_collection.delete_one({"username": username})
        user_logs_collection.delete_many({"username": username})


def authenticate_user(username: str, password: str):
    if not client:
        st.error("MongoDB is not connected. Please check your mongo_uri in secrets.toml or environment variables.")
        return False

    try:
        user = users_collection.find_one({"username": username})
        if user and verify_password(password, user["password"]):
            st.session_state['username'] = user['username']
            st.session_state['full_name'] = user.get('full_name', '')
            st.session_state['profile_pic'] = user.get('profile_pic_base64', '')
            st.session_state['contact_number'] = user.get('contact_number', '')
            st.session_state['authenticated'] = True
            return True
    except Exception as e:
        st.error(f"Login failed due to database error: {e}")
    return False

def add_log_entry(username, entry, images_b64=None, videos_b64=None):
    if client:
        log_data = {
            "username": username,
            "entry": entry,
            "timestamp": datetime.datetime.utcnow(),
            "images_b64": images_b64 or [],
            "videos_b64": videos_b64 or []
        }
        user_logs_collection.insert_one(log_data)
        return True
    return False

def get_user_logs(username):
    if client:
        return user_logs_collection.find({"username": username}).sort("timestamp", -1)
    return []

def delete_log_entry(log_id, username):
    if client:
        result = user_logs_collection.delete_one({"_id": ObjectId(log_id), "username": username})
        return result.deleted_count > 0
    return False

# --- New Forum Database Functions ---
def add_forum_post(username, full_name, title, content, image_b64=None):
    if client:
        post_data = {
            "username": username,
            "full_name": full_name,
            "title": title,
            "content": content,
            "image_b64": image_b64,
            "timestamp": datetime.datetime.utcnow(),
            "likes": []
        }
        forum_posts_collection.insert_one(post_data)
        return True
    return False

def get_all_forum_posts():
    if client:
        return list(forum_posts_collection.find().sort("timestamp", -1))
    return []

def toggle_like_post(post_id, username):
    if client:
        post = forum_posts_collection.find_one({"_id": ObjectId(post_id)})
        if post:
            if username in post['likes']:
                forum_posts_collection.update_one({"_id": ObjectId(post_id)}, {"$pull": {"likes": username}})
            else:
                forum_posts_collection.update_one({"_id": ObjectId(post_id)}, {"$addToSet": {"likes": username}})
            return True
    return False

def add_forum_reply(post_id, username, full_name, content):
    if client:
        reply_data = {
            "post_id": ObjectId(post_id),
            "username": username,
            "full_name": full_name,
            "content": content,
            "timestamp": datetime.datetime.utcnow()
        }
        forum_replies_collection.insert_one(reply_data)
        return True
    return False

def get_replies_for_post(post_id):
    if client:
        return list(forum_replies_collection.find({"post_id": ObjectId(post_id)}).sort("timestamp", 1))
    return []

# =====================================================================================
# 4. APP CONFIGURATION AND DATA/MODEL LOADING
# =====================================================================================

@st.cache_data
def load_and_update_data(filepath):
    df = pd.read_csv(filepath)
    if 'Fertiliser' not in df.columns or 'Duration (Days)' not in df.columns:
        new_data = {
            "Paddy (Rice) - Virippu": {"Fertiliser": "70:35:35", "Duration (Days)": "110-120"},
            "Paddy (Rice) - Mundakan": {"Fertiliser": "90:45:45", "Duration (Days)": "120-140"},
            "Paddy (Rice) - Puncha": {"Fertiliser": "70:35:35", "Duration (Days)": "90-110"},
            "Coconut": {"Fertiliser": "500:320:1200 g/plant", "Duration (Days)": "Perennial"},
            "Rubber": {"Fertiliser": "NPKMg 12:12:12:5", "Duration (Days)": "Perennial"},
            "Banana (Plantain)": {"Fertiliser": "190:110:300 g/plant", "Duration (Days)": "270-365"},
            "Tapioca (Cassava)": {"Fertiliser": "50:50:100", "Duration (Days)": "240-300"},
            "Black Pepper": {"Fertiliser": "100:40:140 g/vine", "Duration (Days)": "Perennial"},
            "Cardamom": {"Fertiliser": "75:75:150", "Duration (Days)": "Perennial"},
            "Tea (high ranges)": {"Fertiliser": "Varies by soil", "Duration (Days)": "Perennial"},
            "Coffee (Robusta/Arabica)": {"Fertiliser": "120:90:120", "Duration (Days)": "Perennial"},
            "Maize (Kharif)": {"Fertiliser": "120:60:40", "Duration (Days)": "90-110"},
            "Tomato": {"Fertiliser": "75:40:25", "Duration (Days)": "90-120"},
            "Okra (Lady's finger)": {"Fertiliser": "75:40:25", "Duration (Days)": "60-70"},
            "Brinjal (Eggplant)": {"Fertiliser": "100:50:50", "Duration (Days)": "120-140"},
            "Cauliflower": {"Fertiliser": "120:60:60", "Duration (Days)": "90-120"},
            "Cabbage": {"Fertiliser": "150:80:80", "Duration (Days)": "80-100"},
            "Pumpkin": {"Fertiliser": "60:80:80", "Duration (Days)": "90-120"},
        }
        df['Fertiliser'] = df['Crop'].map(lambda x: new_data.get(x, {}).get('Fertiliser', 'N/A'))
        df['Duration (Days)'] = df['Crop'].map(lambda x: new_data.get(x, {}).get('Duration (Days)', 'N/A'))
        df.to_csv(filepath, index=False)
    return df

@st.cache_resource
def load_crop_model():
    with open('crop_recommendation_model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_yield_prediction_assets():
    with open('yield_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        le_state, le_crop, le_season = pickle.load(f)
    df_yield = pd.read_csv('kerala_yield_prediction_dataset.csv')
    unique_crops = sorted(df_yield['Crop'].unique())
    unique_seasons = sorted(df_yield['Season'].unique())
    return model, le_state, le_crop, le_season, unique_crops, unique_seasons

@st.cache_resource
def load_leaf_disease_model():
    try:
        model = tf.keras.models.load_model('leaf_disease_model.h5')
        return model
    except (IOError, FileNotFoundError):
        return None
    
# NEW: Function to load a pre-trained model directly from Keras Applications
@st.cache_resource
def load_soil_model():
    try:
        # Load the MobileNetV2 model with weights pre-trained on ImageNet
        model = MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading model from Keras Applications: {e}")
        return None

def get_weather(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except requests.exceptions.RequestException:
        return None

CROP_MAP = {
    'Paddy': 'Paddy(Dhan)(Common)', 'Rice': 'Paddy(Dhan)(Common)', 'Wheat': 'Wheat',
    'Maize': 'Maize', 'Gram': 'Bengal Gram(Gram)(Raw)', 'Moong(Green Gram)': 'Green Gram (Moong)(Whole)',
    'Rapeseed &Mustard': 'Mustard', 'Soyabean': 'Soyabean', 'Tomato': 'Tomato',
    'Potato': 'Potato', 'Onion': 'Onion', 'Banana': 'Banana',
    'Arhar/Tur': 'Arhar (Tur/Red Gram)(Whole)', 'Lentil': 'Lentil (Masur)(Whole)',
    'Groundnut': 'Groundnut', 'Sunflower': 'Sunflower', 'Cotton(lint)': 'Cotton',
    'Jute': 'Jute', 'Garlic': 'Garlic', 'Ginger': 'Ginger(Dry)', 'Dry chillies': 'Dry Chillies'
}

def get_market_price(crop_name, api_key):
    commodity = CROP_MAP.get(crop_name, crop_name)

    def _average_price_from_records(records):
        total_price = 0
        count = 0
        for record in records:
            price_value = record.get('modal_price') or record.get('modal_price_amt') or record.get('Modal_Price')
            if price_value is None:
                continue
            try:
                total_price += float(str(price_value).replace(',', '').strip())
                count += 1
            except (ValueError, TypeError):
                continue
        return total_price / count if count > 0 else None

    # First try the reliable Agmarknet dataset (older but more stable)
    old_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    old_params = {
        "api-key": api_key,
        "format": "json",
        "filters[commodity]": commodity,
        "limit": 50
    }
    try:
        response = requests.get(old_url, params=old_params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', [])
            average_price = _average_price_from_records(records)
            if average_price is not None:
                return average_price
    except requests.exceptions.RequestException:
        pass

    # Fallback to the newer Data.gov dataset if Agmarknet fails
    new_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    new_params = {
        "api-key": api_key,
        "format": "json",
        "filters[Commodity]": commodity,
        "limit": 100
    }
    try:
        response = requests.get(new_url, params=new_params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', [])
            average_price = _average_price_from_records(records)
            if average_price is not None:
                return average_price
    except requests.exceptions.RequestException:
        pass

    # Additional fallback: Hypothetical Indian fintech API (replace with real API)
    # Example: Using a fictional API from an Indian agricultural platform
    # Note: Replace with actual API details from services like CropIn, Ninjacart, or bank APIs
    fintech_url = "https://api.example-indian-agri-finance.com/prices"  # Placeholder URL
    fintech_params = {
        "crop": commodity,
        "region": "Kerala",  # Assuming Kerala focus
        "api_key": st.secrets.get("indian_fintech_api_key", ""),  # Add to secrets.toml
    }
    try:
        response = requests.get(fintech_url, params=fintech_params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Assume response has 'price' field
            price = data.get('price')
            if price:
                return float(price)
    except requests.exceptions.RequestException:
        pass

    # Final fallback: Static average prices for common Kerala crops (based on historical data)
    # These are approximate average prices per kg in INR for Kerala region
    static_prices = {
        'Rice': 35.0, 'Wheat': 25.0, 'Maize': 18.0, 'Sugarcane': 3.5, 'Cotton': 60.0,
        'Groundnut': 85.0, 'Soyabean': 45.0, 'Sunflower': 50.0, 'Mustard': 55.0, 'Coconut': 25.0,
        'Banana': 15.0, 'Mango': 40.0, 'Orange': 30.0, 'Apple': 120.0, 'Grapes': 50.0,
        'Pineapple': 20.0, 'Papaya': 15.0, 'Tomato': 25.0, 'Potato': 20.0, 'Onion': 15.0,
        'Garlic': 150.0, 'Ginger': 100.0, 'Chillies': 80.0, 'Turmeric': 90.0, 'Coriander': 70.0,
        'Cumin': 200.0, 'Pepper': 400.0, 'Cardamom': 800.0, 'Coffee': 250.0, 'Tea': 180.0,
        'Rubber': 150.0, 'Cashew': 600.0, 'Arecanut': 200.0, 'Tapioca': 12.0, 'Sweet Potato': 18.0
    }
    
    # Try to match the crop name (case insensitive)
    for crop_key, price in static_prices.items():
        if crop_key.lower() in commodity.lower() or commodity.lower() in crop_key.lower():
            return price
    
    return None

def transcribe_audio_api(audio_bytes):
    API_URL = "https://transcribe.whisperapi.com"
    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("text")
    except requests.exceptions.RequestException:
        return None
    return None

# =====================================================================================
# 5. LANDING / AUTHENTICATION PAGE
# =====================================================================================
if not st.session_state.get('authenticated', False):
    # Add background image for landing page
    try:
        with open('BG.jpg', 'rb') as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: #f5f5f5;
        }}
        [data-testid="stAppViewContainer"] > .main {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            padding: 48px 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            margin: 60px auto;
            max-width: 900px;
            color: #1a1a1a !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }}
        [data-testid="stAppViewContainer"] > .main * {{
            color: #1a1a1a !important;
        }}
        h1, h2, h3 {{
            color: #2d5016 !important;
            font-weight: 800 !important;
            text-shadow: none !important;
            margin-bottom: 24px !important;
        }}
        label {{
            color: #3d7a2d !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
            display: block !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }}
        .stTextInput > label {{
            color: #3d7a2d !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
            display: block !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }}
        .stTextInput > div > div > input {{
            font-size: 15px !important;
            border: 2px solid #d0d0d0 !important;
            border-radius: 8px !important;
            background-color: #f9f9f9 !important;
            color: #000000 !important;
            padding: 12px 14px !important;
            transition: all 0.3s ease !important;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: #2d5016 !important;
            background-color: #ffffff !important;
            box-shadow: 0 0 0 3px rgba(45, 80, 22, 0.1) !important;
        }}
        .stSelectbox > div > div > div {{
            font-size: 15px !important;
            color: #000000 !important;
            background-color: #f9f9f9 !important;
            border-radius: 8px !important;
        }}
        .stSelectbox {{
            margin-bottom: 24px !important;
        }}
        .stButton > button {{
            font-weight: 700 !important;
            font-size: 15px !important;
            color: #ffffff !important;
            background: linear-gradient(135deg, #2d5016 0%, #3d7a2d 100%) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 32px !important;
            width: 100% !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(45, 80, 22, 0.2) !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(45, 80, 22, 0.3) !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #555555 !important;
            font-weight: 600 !important;
            text-shadow: none !important;
            border-bottom: 2px solid transparent !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease !important;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: #2d5016 !important;
            background: transparent !important;
            border-bottom-color: #2d5016 !important;
        }}
        [data-testid="stForm"] {{
            background: transparent !important;
            border: none !important;
            padding: 20px 0 !important;
        }}
        .stForm > button {{
            margin-top: 16px !important;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(10px) !important;
        }}
        .landing-hero {{
            text-align: center !important;
            padding: 40px 0 !important;
        }}
        .landing-section {{
            margin: 40px 0 !important;
            padding: 30px !important;
            background: rgba(45, 80, 22, 0.05) !important;
            border-radius: 12px !important;
            border-left: 5px solid #2d5016 !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load background image: {e}")
    
    # Initialize landing page state
    if 'show_landing' not in st.session_state:
        st.session_state['show_landing'] = True

    # If a direct navigation request arrives from the floating chatbot icon, leave the landing page
    nav_param = st.query_params.get('nav') if hasattr(st, 'query_params') else None
    if isinstance(nav_param, list):
        nav_param = nav_param[0] if nav_param else None

    # Landing Page
    if st.session_state.get('show_landing', True):
        # Add background image for landing page only
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url('data:image/png;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class='landing-hero'>
                <h1 style='font-size: 48px; margin-bottom: 16px; color: #2d5016 !important;'>🌿 KrishiSakhi</h1>
                <p style='font-size: 18px; color: #2d5016; font-weight: 600;'>Your Comprehensive Agricultural Companion</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # About Section
        st.markdown("""
        <style>
        .page-hero {
            border-radius: 28px;
            background: linear-gradient(135deg, #2b4f17 0%, #6ea03d 100%);
            padding: 42px 32px;
            color: #f1f7e7;
            position: relative;
            overflow: hidden;
            box-shadow: 0 30px 80px rgba(17, 53, 16, 0.28);
            margin-bottom: 28px;
        }
        .page-hero::before {
            content: '';
            position: absolute;
            width: 340px;
            height: 340px;
            top: -60px;
            right: -80px;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 50%;
            filter: blur(24px);
        }
        .landing-hero h1 {
            font-size: 3.1rem;
            margin: 0 0 10px;
            line-height: 1.02;
            color: #2d5016 !important;
        }
        .page-hero h1 {
            font-size: 3.1rem;
            margin: 0 0 10px;
            line-height: 1.02;
            color: #ffffff !important;
        }
        .page-hero p {
            font-size: 1.12rem;
            line-height: 1.75;
            max-width: 760px;
            margin: 0;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            padding: 12px 18px;
            margin-top: 20px;
            color: #f1f7e7;
            font-weight: 600;
            backdrop-filter: blur(8px);
        }
        .hero-pill-group {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 22px;
        }
        .page-block {
            background: #f8fbf0;
            border: 1px solid #e5eed6;
            border-radius: 26px;
            padding: 30px 28px;
            box-shadow: 0 24px 50px rgba(33, 72, 24, 0.08);
            margin-bottom: 24px;
        }
        .page-block h2 {
            margin-top: 0;
            color: #254523;
        }
        .page-block p {
            color: #41513c;
            line-height: 1.8;
            font-size: 1rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }
        .stat-card {
            background: #ffffff;
            border-radius: 22px;
            padding: 24px 20px;
            border: 1px solid #e6edd7;
            box-shadow: 0 24px 40px rgba(45, 80, 22, 0.08);
            text-align: center;
        }
        .stat-card .value {
            color: #2d5016;
            font-size: 2.45rem;
            font-weight: 800;
            line-height: 1;
        }
        .stat-card .label {
            margin-top: 12px;
            color: #5d6e55;
            font-size: 0.95rem;
        }
        .contact-panel {
            background: #ffffff;
            border-radius: 24px;
            border: 1px solid #e6eed7;
            padding: 28px 24px;
            box-shadow: 0 20px 45px rgba(45, 80, 22, 0.08);
            margin-bottom: 36px;
        }
        .contact-panel h2 {
            margin: 0 0 12px;
            color: #254523;
        }
        .contact-panel p {
            margin: 0 0 16px;
            color: #4e5f46;
            line-height: 1.75;
        }
        .contact-row {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
            color: #435741;
            font-size: 1rem;
        }
        .contact-row span {
            font-size: 1.3rem;
        }
        .contact-row a {
            color: #2d5016;
            text-decoration: none;
            font-weight: 600;
        }
        .contact-row a:hover {
            text-decoration: underline;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 24px;
        }
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f0f8e8 100%);
            border: 1px solid #e6edd7;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 20px 40px rgba(45, 80, 22, 0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 28px 60px rgba(45, 80, 22, 0.15);
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 16px;
        }
        .feature-card h3 {
            margin: 0 0 16px;
            color: #254523;
            font-size: 1.25rem;
        }
        .feature-card ul {
            margin: 0;
            padding-left: 20px;
            color: #4e5f46;
            line-height: 1.7;
        }
        .feature-card li {
            margin-bottom: 8px;
        }
        .contact-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 24px;
        }
        .contact-item {
            background: linear-gradient(135deg, #ffffff 0%, #f0f8e8 100%);
            border: 1px solid #e6edd7;
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 16px 32px rgba(45, 80, 22, 0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .contact-item:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 48px rgba(45, 80, 22, 0.12);
        }
        .contact-icon {
            font-size: 2rem;
            margin-bottom: 12px;
        }
        .contact-item h4 {
            margin: 0 0 8px;
            color: #254523;
            font-size: 1.1rem;
        }
        .contact-item p {
            margin: 0;
            color: #4e5f46;
        }
        .contact-item a {
            color: #2d5016;
            text-decoration: none;
            font-weight: 600;
        }
        .contact-item a:hover {
            text-decoration: underline;
        }
        .page-hero p {
            color: #ffffff;
        }
        </style>
        <div class='page-hero'>
            <h1 style='color: #2d5016 !important;'>🌿 KrishiSakhi</h1>
            <p>
            System of Agri-information Resources Auto-transmission and Technology Hub Interface, powered by KrishiSakhi and team. KrishiSakhi delivers a seamless multimedia link between farmers, modern agriculture technologies, expert guidance, and local farming knowledge.
            </p>
            <div class='hero-pill-group'>
                <div class='hero-pill'>Farmer-first design</div>
                <div class='hero-pill'>Local agriculture support</div>
                <div class='hero-pill'>National reach</div>
            </div>
        </div>
        <div class='page-block'>
            <h2>📖 About KrishiSakhi</h2>
            <p>
            “KrishiSakhi” is an ICT-based interface solution with the ultimate goal of providing an intelligent online platform for supporting agriculture at the local niche level with national perspective. It aims to connect farmers with the latest technologies, knowledge base, and a pool of subject matter experts.
            </p>
            <div class='stats-grid'>
                <div class='stat-card'>
                    <div class='value'>27,564,233</div>
                    <div class='label'>Farmers Registered</div>
                </div>
                <div class='stat-card'>
                    <div class='value'>751</div>
                    <div class='label'>Registered Institutions</div>
                </div>
                <div class='stat-card'>
                    <div class='value'>344,094</div>
                    <div class='label'>Villages Covered</div>
                </div>
            </div>
        </div>
        <div class='contact-panel'>
            <h2>🌾 Our Farmer Motto</h2>
            <p>Growing together with knowledge, care, and innovation — empowering every field, family, and future.</p>
            <div class='contact-row'><span>🌱</span>We stand for smart farming, stronger communities, and sustainable harvests.</div>
            <div class='contact-row'><span>🤝</span>Your success is our mission — today, tomorrow, and every season ahead.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Section
        st.markdown("""
        <div class='page-block'>
            <h2>✨ What We Offer</h2>
            <div class='feature-grid'>
                <div class='feature-card'>
                    <div class='feature-icon'>🌾</div>
                    <h3>Crop Intelligence</h3>
                    <ul>
                        <li>Smart crop recommendations based on soil & weather</li>
                        <li>Yield forecasting and production planning</li>
                        <li>Optimal sowing and harvest calendars</li>
                    </ul>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>🧬</div>
                    <h3>Plant Health Monitoring</h3>
                    <ul>
                        <li>AI-powered disease detection</li>
                        <li>Pest identification and management</li>
                        <li>Real-time crop health analytics</li>
                    </ul>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>🌍</div>
                    <h3>Environmental Data</h3>
                    <ul>
                        <li>Live weather forecasting</li>
                        <li>Soil type and property analysis</li>
                        <li>Climate-specific farming guidance</li>
                    </ul>
                </div>
                <div class='feature-card'>
                    <div class='feature-icon'>🤝</div>
                    <h3>Community Support</h3>
                    <ul>
                        <li>Farmer-to-farmer knowledge sharing</li>
                        <li>Expert guidance and resources</li>
                        <li>Multilingual access (8+ Indian languages)</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Call to Action Buttons
        st.markdown("""
        <div class='page-block'>
            <h2>📞 Connect With Us</h2>
            <div class='contact-grid'>
                <div class='contact-item'>
                    <div class='contact-icon'>📧</div>
                    <h4>Email</h4>
                    <p><a href='mailto:krishisakhi@gmail.com'>krishisakhi@gmail.com</a></p>
                </div>
                <div class='contact-item'>
                    <div class='contact-icon'>📱</div>
                    <h4>Phone</h4>
                    <p><a href='tel:+916297648920'>+91 62976 48920</a></p>
                </div>
                <div class='contact-item'>
                    <div class='contact-icon'>🏢</div>
                    <h4>Organization</h4>
                    <p>KrishiSakhi Team</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Call to Action Buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            pass
        with col2:
            st.markdown(
                """
                <style>
                [data-testid="stColumns"] > div:nth-child(2) button {
                    color: white !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🚀 Get Started - Login / Sign Up", key="start_auth", use_container_width=True):
                st.session_state['show_landing'] = False
                st.rerun()
        with col3:
            pass
    
    # Login/Signup Section (shown after landing page)
    else:
        # Remove background image for login section
        st.markdown("""
        <style>
        body {
            background-image: none !important;
            background-color: #f5f5f5 !important;
        }
        [data-testid="stAppViewContainer"] {
            background-image: none !important;
            background-color: #f5f5f5 !important;
        }
        [data-testid="stAppViewContainer"]::before {
            background-image: none !important;
        }
        [data-testid="stVerticalBlock"] > div.stButton button {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Home", key="back_to_landing"):
            st.session_state['show_landing'] = True
            st.rerun()
        
        st.markdown("---")
        
        with st.sidebar:
            st.subheader("Login Required")
            st.markdown("**Please login or sign up to access KrishiSakhi's full features.**")
            if not client:
                st.error("MongoDB is not connected. Check your mongodb_uri in secrets.toml.")
        
        st.title(t('login_page_title'))
        # Ensure widget state matches current language selection
        if 'login_lang_selector' not in st.session_state:
            st.session_state['login_lang_selector'] = st.session_state.get('lang', 'en')

        st.selectbox(
            label=t('language_selector_label'),
            options=['en', 'hi', 'ml', 'ta', 'te', 'bn', 'pa', 'mr'],
            format_func=lambda x: {'en': 'English', 'hi': 'हिन्दी', 'ml': 'മലയാളം', 'ta': 'தமிழ்', 'te': 'తెలుగు', 'bn': 'বাংলা', 'pa': 'ਪੰਜਾਬੀ', 'mr': 'मराठी'}[x],
            key='login_lang_selector',
            on_change=set_login_language
        )
        login_tab, signup_tab = st.tabs([t('login_tab'), t('signup_tab')])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input(t('username_label'))
                password = st.text_input(t('password_label'), type="password")
                submitted = st.form_submit_button(t('login_button'))
                if submitted:
                    if not client:
                        st.error("MongoDB connection is unavailable. Please check your mongo_uri in secrets.toml.")
                    elif authenticate_user(username, password):
                        st.success("Login successful. Loading app...")
                        st.rerun()
                    else:
                        st.error(t('login_error'))
        with signup_tab:
            with st.form("signup_form"):
                full_name = st.text_input(t('full_name_label'), key="signup_fullname")
                new_username = st.text_input(t('username_label'), key="signup_username")
                new_password = st.text_input(t('password_label'), type="password", key="signup_password")
                submitted = st.form_submit_button(t('signup_button'))
                if submitted:
                    if not full_name or not new_username or not new_password:
                        st.warning("All fields are required.")
                    elif user_exists(new_username):
                        st.error(t('signup_error_exists'))
                    else:
                        try:
                            if add_user(new_username, new_password, full_name):
                                st.success(t('signup_success'))
                            else:
                                st.error(t('signup_error_general'))
                        except Exception as e:
                            st.error(f"{t('signup_error_general')}: {e}")
# =====================================================================================
# 6. MAIN APPLICATION
# =====================================================================================
else:
    # Load models and data
    try:
        crop_model = load_crop_model()
    except Exception as e:
        st.error(f"Failed to load crop model: {e}")
        crop_model = None
    
    try:
        yield_model, le_state, le_crop, le_season, unique_crops, unique_seasons = load_yield_prediction_assets()
    except Exception as e:
        st.error(f"Failed to load yield prediction assets: {e}")
        yield_model = None
        le_state = le_crop = le_season = unique_crops = unique_seasons = None
    
    try:
        leaf_disease_model = load_leaf_disease_model()
    except Exception as e:
        st.error(f"Failed to load leaf disease model: {e}")
        leaf_disease_model = None
    
    try:
        kerala_crops_df = load_and_update_data('kerala_crops.csv')
    except Exception as e:
        st.error(f"Failed to load crop data: {e}")
        kerala_crops_df = None
    
    try:
        soil_model = load_soil_model()
    except Exception as e:
        st.error(f"Failed to load soil model: {e}")
        soil_model = None

    # If a navigation request was made by buttons or URL query, apply it before creating the page selector widget
    def _apply_requested_page():
        page_keymap = {
            "Home": "page_home", "Profile": "page_profile", "Logs": "page_logs", "Crop Recommendation": "page_crop_recommendation",
            "Yield Prediction": "page_yield_prediction", "Crop Calendar": "page_crop_calendar", "Crop Guide": "page_crop_guide",
            "Weather Report": "page_weather_report", "Leaf Disease Prediction": "page_leaf_disease_prediction",
            "Pest Prediction": "page_pest_prediction", "Soil Type Prediction": "page_soil_type_prediction",
            "Translator": "page_translator", "Community Forum": "page_community_forum"
        }

        # Priority: explicit request in session state
        if st.session_state.get('_requested_page'):
            requested = st.session_state.pop('_requested_page')
            st.session_state['page_selector'] = page_keymap.get(requested, requested)
            return

        # Fallback: query param (for floating button)
        params = st.query_params
        nav = params.get('nav')
        if isinstance(nav, list):
            nav = nav[0] if nav else None
        if nav:
            st.session_state['page_selector'] = page_keymap.get(nav, nav)

        # Also normalise any existing selector value (for backward compatibility with older versions)
        current = st.session_state.get('page_selector')
        if current in page_keymap:
            st.session_state['page_selector'] = page_keymap[current]

    if 'page_selector' not in st.session_state:
        st.session_state['page_selector'] = 'page_home'

    _apply_requested_page()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🌿 KrishiSakhi")
        
        profile_pic_data = st.session_state.get('profile_pic', '')
        if profile_pic_data:
            try:
                st.image(base64.b64decode(profile_pic_data), use_container_width=True, caption=st.session_state.get('full_name', ''))
            except Exception:
                 st.markdown("""<div style="display: flex; justify-content: center;"><svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="display: flex; justify-content: center;"><svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg></div>""", unsafe_allow_html=True)
        st.info(t('welcome_back', username=st.session_state.get('full_name', 'Farmer')))
        
        # Keep sidebar language selector synced with the chosen app language
        if 'lang_selector' not in st.session_state:
            st.session_state['lang_selector'] = st.session_state.get('lang', 'en')

        def set_language():
            st.session_state['lang'] = st.session_state['lang_selector']
            st.session_state['login_lang_selector'] = st.session_state['lang_selector']
            st.session_state.chat_history = []

        st.selectbox(
            label=t('language_selector_label'),
            options=['en', 'hi', 'ml', 'ta', 'te', 'bn', 'pa', 'mr'],
            format_func=lambda x: {'en': 'English', 'hi': 'हिन्दी', 'ml': 'മലയാളം', 'ta': 'தமிழ்', 'te': 'తెలుగు', 'bn': 'বাংলা', 'pa': 'ਪੰਜਾਬੀ', 'mr': 'मराठी'}[x],
            key='lang_selector',
            on_change=set_language
        )
        
        PAGES_KEYS = {
            'page_home': "Home", 'page_profile': "Profile", 'page_logs': "Logs", 'page_crop_recommendation': "Crop Recommendation",
            'page_yield_prediction': "Yield Prediction", 'page_crop_calendar': "Crop Calendar", 'page_crop_guide': "Crop Guide",
            'page_weather_report': "Weather Report", 'page_leaf_disease_prediction': "Leaf Disease Prediction",
            'page_pest_prediction': "Pest Prediction", 'page_soil_type_prediction': "Soil Type Prediction",
            'page_translator': "Translator", 'page_community_forum': "Community Forum"
        }

        # Use internal page IDs for navigation, but display translated labels
        pages = list(PAGES_KEYS.keys())
        def format_page(key):
            return t(key)

        selected_page = st.radio(t('go_to'), pages, format_func=format_page, key="page_selector")
        page = selected_page

        st.divider()
        st.subheader(t('feedback_header'))
        rating = st.slider(t('rating_label'), 1, 5, 5, label_visibility="collapsed", format="%d ⭐")
        if st.button(t('submit_button')):
            st.success(t('feedback_success'))
        
        st.divider()
        st.subheader(t('contact_header'))
        st.markdown("""
        # **KrishiSakhi**
        📧 krishisakhi@gmail.com
                    
        📞 +91 6297648920
        """)
        
        st.divider()
        if st.button(t('logout_button')):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- PAGE CONTENT ---

    def render_top_nav_buttons():
        buttons = [
            ("page_home", t('page_home')),
            ("page_crop_recommendation", t('page_crop_recommendation')),
            ("page_yield_prediction", t('page_yield_prediction')),
            ("page_weather_report", t('page_weather_report')),
            ("page_leaf_disease_prediction", t('page_leaf_disease_prediction')),
            ("page_community_forum", t('page_community_forum')),
        ]
        cols = st.columns(len(buttons), gap="small")
        for (page_key, label), col in zip(buttons, cols):
            if col.button(label, key=f"nav_{page_key}"):
                st.session_state['_requested_page'] = page_key
                st.rerun()

    render_top_nav_buttons()

    # --- FLOATING AI CHATBOT WIDGET ---
    try:
        with open("ICON.jpg", "rb") as img_file:
            chatbot_icon_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        chatbot_icon_base64 = None

    chatbot_icon_html = (
        f'<img src="data:image/jpeg;base64,{chatbot_icon_base64}" style="width: 34px; height: 34px; object-fit: cover; border-radius: 50%;" />'
        if chatbot_icon_base64 else
        '💬'
    )

    st.markdown(f"""
        <div class="chat-widget">
            <input type="checkbox" id="chat-toggle" class="chat-toggle-checkbox">
            <label for="chat-toggle" class="chat-launcher" title="AI Chatbot">
                {chatbot_icon_html}
            </label>
            <div class="chat-panel">
                <div class="chat-panel-header">
                    <div>AI Chatbot</div>
                    <label for="chat-toggle">✕</label>
                </div>
                <div class="chat-panel-body">
                    <iframe src="https://chat.cxgenie.ai?aid=71a90137-4bcc-490a-bdce-5d0d8c894d5b&lang=en"></iframe>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    if page == "page_home":

        # Hero section
        st.markdown(
            f"""
            <div class='hero'>
                <h1>{t('home_title')}</h1>
                <p>{t('home_subtitle')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Use tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs([t('home_tab_crop_management'), t('home_tab_analytics'), t('home_tab_environment'), t('home_tab_community')])

        with tab1:
            st.markdown(t('home_section_crop_planning'))
            card_cols = st.columns(2, gap="large")
            with card_cols[0]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🌾 {t('page_crop_recommendation')}</h4>
                        <p>{t('quick_action_crop_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_crop"):
                    st.session_state['_requested_page'] = "page_crop_recommendation"
                    st.rerun()

            with card_cols[1]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🗓 {t('page_crop_calendar')}</h4>
                        <p>{t('home_card_crop_calendar_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_calendar"):
                    st.session_state['_requested_page'] = "page_crop_calendar"
                    st.rerun()

        with tab2:
            st.markdown(t('home_section_data_insights'))
            card_cols = st.columns(2, gap="large")
            with card_cols[0]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>📈 {t('page_yield_prediction')}</h4>
                        <p>{t('home_card_yield_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_yield"):
                    st.session_state['_requested_page'] = "page_yield_prediction"
                    st.rerun()

            with card_cols[1]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🏞 {t('page_soil_type_prediction')}</h4>
                        <p>{t('home_card_soil_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_soil"):
                    st.session_state['_requested_page'] = "page_soil_type_prediction"
                    st.rerun()

        with tab3:
            st.markdown(t('home_section_environment'))
            card_cols = st.columns(2, gap="large")
            with card_cols[0]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🌦 {t('page_weather_report')}</h4>
                        <p>{t('quick_action_weather_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_weather"):
                    st.session_state['_requested_page'] = "page_weather_report"
                    st.rerun()

            with card_cols[1]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🌿 {t('page_leaf_disease_prediction')}</h4>
                        <p>{t('leaf_disease_intro')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_disease"):
                    st.session_state['_requested_page'] = "page_leaf_disease_prediction"
                    st.rerun()

        with tab4:
            st.markdown(t('home_section_community'))
            card_cols = st.columns(2, gap="large")
            with card_cols[0]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🐛 {t('page_pest_prediction')}</h4>
                        <p>{t('home_card_pest_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_pest"):
                    st.session_state['_requested_page'] = "page_pest_prediction"
                    st.rerun()

            with card_cols[1]:
                st.markdown(
                    f"""
                    <div class='feature-card'>
                        <h4>🌐 {t('page_translator')}</h4>
                        <p>{t('home_card_translator_desc')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(t('go_button'), key="home_nav_translator"):
                    st.session_state['_requested_page'] = "page_translator"
                    st.rerun()

        st.divider()
        num_languages = len(translations)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label=t('languages_supported_label'), value=num_languages)
        col2.metric(label=t('farmers_helped_label'), value="1,500+")
        col3.metric(label=t('crops_covered_label'), value="20+")
        col4.metric(label=t('recommendations_given_label'), value="10,000+")
        st.divider()
        
        with st.expander(t('home_about_header')):
            st.markdown(t('home_intro'))

    elif page == "page_profile":
        st.header(t('profile_header'))
        st.write(t('profile_subheader'))

        with st.form("profile_form"):
            full_name = st.text_input(t('full_name_label'), value=st.session_state.get('full_name', ''))
            contact_number = st.text_input(t('contact_number_label'), value=st.session_state.get('contact_number', ''))
            new_username = st.text_input(t('new_username_label'), value=st.session_state.get('username', ''))
            uploaded_pic = st.file_uploader(t('profile_pic_label'), type=['png', 'jpg', 'jpeg'])
            new_password = st.text_input(t('new_password_label'), type="password", placeholder="Enter new password (optional)")
            confirm_password = st.text_input(t('confirm_password_label'), type="password", placeholder="Confirm new password")

            submitted = st.form_submit_button(t('update_profile_button'))

            if submitted:
                updates = {}
                if new_password and new_password != confirm_password:
                    st.error(t('password_mismatch_error'))
                else:
                    if full_name != st.session_state.get('full_name', ''):
                        updates['full_name'] = full_name
                    if contact_number != st.session_state.get('contact_number', ''):
                        updates['contact_number'] = contact_number
                    if new_username != st.session_state.get('username', ''):
                        updates['username'] = new_username
                    if new_password:
                        updates['password'] = new_password
                    if uploaded_pic is not None:
                        bytes_data = uploaded_pic.getvalue()
                        updates['profile_pic_base64'] = base64.b64encode(bytes_data).decode()

                    if not updates:
                        st.info("No changes were made.")
                    else:
                        if update_user_profile(st.session_state['username'], updates):
                            st.success(t('profile_update_success'))
                            if 'username' in updates:
                                st.session_state['username'] = updates['username']
                            if 'full_name' in updates:
                                st.session_state['full_name'] = updates['full_name']
                            if 'contact_number' in updates:
                                st.session_state['contact_number'] = updates['contact_number']
                            if 'profile_pic_base64' in updates:
                                st.session_state['profile_pic'] = updates['profile_pic_base64']
                            st.rerun()
                        else:
                            st.error(t('profile_update_error'))

        # --- SMS Contact Configuration Section ---
        st.divider()
        st.subheader("📱 SMS Notification Settings")
        
        user_details = get_user_details(st.session_state['username'])
        current_contact = user_details.get('contact_number', '') if user_details else ''
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(
                "📲 **Enable SMS Reminders**\n\n"
                "Add your phone number below to receive SMS reminders for:\n"
                "- 🌾 Crop sowing windows\n"
                "- 🎯 Harvest timings\n"
                "- 🧪 Fertiliser schedules"
            )
        
        with col2:
            if current_contact:
                st.success(f"✅ SMS Enabled\n\n**{current_contact}**")
            else:
                st.warning("❌ SMS Not Configured")
        
        st.write("**Enter your 10-digit phone number or with country code (+91):**")
        
        col_sms1, col_sms2, col_sms3 = st.columns([2, 1, 1])
        with col_sms1:
            sms_contact = st.text_input(
                "Phone Number for SMS",
                value=current_contact,
                placeholder="+91XXXXXXXXXX or XXXXXXXXXX",
                key="sms_contact_input"
            )
        
        with col_sms2:
            update_sms = st.button("💾 Save", key="save_sms_contact")
        
        with col_sms3:
            if current_contact:
                test_sms = st.button("📤 Test SMS", key="test_sms_button")
            else:
                test_sms = False
        
        # Validate and save SMS contact
        if update_sms:
            if not sms_contact:
                st.error("❌ Please enter a phone number")
            elif len(sms_contact.replace("+", "").replace(" ", "").replace("-", "")) < 10:
                st.error("❌ Phone number must be at least 10 digits")
            else:
                if update_user_profile(st.session_state['username'], {'contact_number': sms_contact}):
                    st.success(f"✅ SMS contact updated: {sms_contact}")
                    st.session_state['contact_number'] = sms_contact
                    st.rerun()
                else:
                    st.error("❌ Failed to update SMS contact")
        
        # Test SMS functionality
        if test_sms:
            with st.spinner("📤 Sending test SMS..."):
                test_message = "KrishiSakhi Test: Your SMS notifications are working! 🌾"
                if send_sms(current_contact, test_message):
                    st.success("✅ Test SMS sent successfully! Check your phone.")
                else:
                    st.error("❌ Failed to send test SMS. Please check your API key and try again.")

        st.divider()
        st.subheader(t('danger_zone_header'))
        st.error(t('delete_account_warning'))

        if st.button(t('delete_account_button'), type="secondary", key="initiate_delete"):
            st.session_state['delete_pressed'] = True

        if st.session_state.get('delete_pressed'):
            confirmation = st.text_input(t('delete_confirmation_label'))
            if st.button(t('delete_account_confirm_button'), type="primary"):
                if confirmation == "DELETE":
                    delete_user(st.session_state['username'])
                    st.success(t('account_deleted_success'))
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                else:
                    st.warning(t('incorrect_delete_confirmation'))

    elif page == "page_logs":
        st.header(t('logs_header'))
        st.write(t('logs_intro'))

        if 'log_input_text' not in st.session_state:
            st.session_state.log_input_text = ""

        st.write(t('record_log_prompt'))
        
        class AudioProcessor(AudioProcessorBase):
            def __init__(self) -> None:
                self.audio_buffer = io.BytesIO()
                self.lock = threading.Lock()

            def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                with self.lock:
                    self.audio_buffer.write(frame.to_ndarray().tobytes())
                return frame

        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"video": False, "audio": True},
        )
        
        if not webrtc_ctx.state.playing and st.session_state.get("audio_buffer"):
            with st.spinner(t('transcribing_audio')):
                audio_buffer = st.session_state.get("audio_buffer")
                if audio_buffer:
                    try:
                        # Convert raw PCM audio to a WAV file in memory
                        audio_buffer.seek(0)
                        raw_data = audio_buffer.read()
                        
                        wav_io = io.BytesIO()
                        with wave.open(wav_io, 'wb') as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000) # 16k sample rate
                            wf.writeframes(raw_data)
                        wav_io.seek(0)

                        recognized_text = transcribe_audio_api(wav_io.getvalue())
                        
                        if recognized_text:
                            st.session_state.log_input_text += recognized_text + " "
                        else:
                            st.warning(t('transcription_error'))
                        
                        st.session_state["audio_buffer"] = None # Clear buffer
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred during audio processing: {e}")

        if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
            st.session_state["audio_buffer"] = webrtc_ctx.audio_processor.audio_buffer

        with st.form("log_form"):
            log_entry_text = st.text_area(t('new_log_label'), value=st.session_state.log_input_text, height=150, key="log_text_area")
            
            uploaded_files = st.file_uploader(
                t('upload_media_label'),
                type=['png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'],
                accept_multiple_files=True
            )

            submitted = st.form_submit_button(t('save_log_button'))
            if submitted:
                images_b64 = []
                videos_b64 = []
                
                for uploaded_file in uploaded_files:
                    bytes_data = uploaded_file.read()
                    b64_data = base64.b64encode(bytes_data).decode()
                    mime_type = uploaded_file.type

                    if "image" in mime_type:
                        images_b64.append(f"data:{mime_type};base64,{b64_data}")
                    elif "video" in mime_type:
                        videos_b64.append(f"data:{mime_type};base64,{b64_data}")

                if log_entry_text or images_b64 or videos_b64:
                    if add_log_entry(st.session_state['username'], log_entry_text, images_b64=images_b64, videos_b64=videos_b64):
                        st.session_state.log_input_text = "" # Clear text on save
                        st.success(t('log_saved_success'))
                        st.rerun()
                    else:
                        st.error("Failed to save log entry.")
        
        st.divider()
        st.subheader(t('past_logs_header'))
        
        logs = list(get_user_logs(st.session_state['username']))
        if logs:
            for log in logs:
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        utc = timezone('UTC')
                        ist = timezone('Asia/Kolkata')
                        utc_time = log['timestamp'].replace(tzinfo=utc)
                        ist_time = utc_time.astimezone(ist)
                        st.caption(ist_time.strftime("%A, %B %d, %Y at %I:%M %p IST"))
                        if log.get("entry"):
                            st.write(log['entry'])
                        
                        if log.get("images_b64"):
                            st.write(f"**{t('attached_images')}**")
                            for img_b64 in log.get("images_b64"):
                                st.image(img_b64)
                        
                        if log.get("videos_b64"):
                            st.write(f"**{t('attached_videos')}**")
                            for vid_b64 in log.get("videos_b64"):
                                st.video(vid_b64)

                    with col2:
                        if st.button(t('delete_log_button'), key=f"delete_{log['_id']}", type="secondary"):
                            delete_log_entry(str(log['_id']), st.session_state['username'])
                            st.toast(t('log_deleted_success'))
                            st.rerun()
        else:
            st.info(t('no_logs_message'))

    elif page == "page_crop_recommendation":
        st.header(t('crop_rec_header'))
        st.write(t('crop_rec_intro'))
        
        if 'last_crop_recommendation' not in st.session_state:
            st.session_state['last_crop_recommendation'] = None

        col1, col2, col3 = st.columns(3)
        with col1:
            n = st.slider(t('nitrogen_label'), 0, 140, 90)
            p = st.slider(t('phosphorus_label'), 5, 145, 42)
            k = st.slider(t('potassium_label'), 5, 205, 43)
        with col2:
            temp = st.slider(t('temperature_label'), 8.0, 44.0, 20.88, format="%.2f")
            humidity = st.slider(t('humidity_label'), 14.0, 100.0, 82.00, format="%.2f")
            ph = st.slider(t('ph_label'), 3.5, 9.9, 6.50, format="%.2f")
        with col3:
            rainfall = st.slider(t('rainfall_label'), 20.0, 299.0, 202.94, format="%.2f")
        
        if st.button(t('recommend_crop_button')):
            with st.spinner("Analyzing soil parameters..."):
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                prediction = crop_model.predict(np.array([[n, p, k, temp, humidity, ph, rainfall]]))
                st.session_state['last_crop_recommendation'] = prediction[0]
                progress_bar.empty()

        if st.session_state['last_crop_recommendation']:
            recommended_crop = st.session_state['last_crop_recommendation']
            st.success(t('recommendation_success', crop=translate_if_needed(recommended_crop.title())))
            
            st.divider()
            st.subheader(t('crop_info_header', crop=translate_if_needed(recommended_crop.title())))

            crop_data = CROP_INFO.get(recommended_crop.lower())
            if crop_data:
                with st.expander("📖 Detailed Crop Information", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{t('description_heading')}**")
                        st.write(translate_if_needed(crop_data['description']))
                        
                        st.markdown(f"**{t('irrigation_heading')}**")
                        st.write(translate_if_needed(crop_data['irrigation']))
                    with col2:
                        st.markdown(f"**{t('conditions_heading')}**")
                        st.write(translate_if_needed(crop_data['conditions']))

                        st.markdown(f"**{t('pests_heading')}**")
                        st.write(translate_if_needed(crop_data['pests']))
            else:
                st.info(f"Detailed information for {recommended_crop.title()} is not yet available.")

    elif page == "page_crop_guide":
        st.header(t('crop_guide_title'))

        try:
            # Load the fertigation and irrigation data from the CSV file
            df_guide = pd.read_csv('kerala_fertigation_irrigation.csv')
        except FileNotFoundError:
            st.error(t('file_not_found_error'))
        else:
            # Create a dropdown for the user to select a crop
            unique_crops = sorted(df_guide['Crop'].unique())
            selected_crop = st.selectbox(t('select_crop_label'), options=unique_crops, format_func=lambda x: translate_if_needed(x), key='guide_crop_selection')

            if selected_crop:
                # Filter the dataframe based on the selected crop
                crop_specific_df = df_guide[df_guide['Crop'] == selected_crop]
                
                # Create a second dropdown for the growth stage, based on the chosen crop
                unique_stages = crop_specific_df['Stage'].unique()
                selected_stage = st.selectbox(t('select_stage_label'), options=unique_stages, format_func=lambda x: t(f'stage_{x.lower()}'), key='guide_stage_selection')

                if selected_stage:
                    # Filter the dataframe again for the selected stage
                    stage_specific_df = crop_specific_df[crop_specific_df['Stage'] == selected_stage]

                    st.subheader(t('guide_table_header'))
                    
                    # Define which columns to show in the final table
                    columns_to_display = [
                        'Days Range', 
                        'Fertilizer', 
                        'Dosage (kg/acre)', 
                        'Frequency', 
                        'Irrigation Depth / Volume', 
                        'Irrigation Frequency'
                    ]
                    
                    # Translate column names
                    translated_columns = [t(f'column_{col.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(".", "")}') for col in columns_to_display]
                    
                    # Rename columns in the dataframe for display
                    display_df = stage_specific_df[columns_to_display].copy()
                    display_df.columns = translated_columns
                    
                    # Translate fertilizer names and other text columns
                    text_columns = ['Fertilizer', 'Frequency', 'Irrigation Frequency']
                    for col in text_columns:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: translate_if_needed(str(x)))
                    
                    # Display the final, filtered data in a clean table
                    st.dataframe(
                        display_df.reset_index(drop=True), 
                        use_container_width=True
                    )

    elif page == "page_yield_prediction":
        st.header(t('yield_pred_header'))
        st.write(t('yield_pred_intro'))
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input(t('year_label'), 2000, 2030, 2025)
            crop = st.selectbox(t('crop_label'), options=unique_crops, format_func=lambda x: translate_if_needed(x))
        with col2:
            season = st.selectbox(t('season_label'), options=unique_seasons, format_func=lambda x: translate_if_needed(x))
            area = st.number_input(t('area_label'), 1.0, value=10.0, format="%.2f")
        
        if st.button(t('predict_yield_button')):
            try:
                features = np.array([[
                    le_state.transform(['Kerala'])[0],
                    le_crop.transform([crop])[0],
                    le_season.transform([season])[0],
                    year, area
                ]])
                total_production = yield_model.predict(features)[0]
                
                predicted_yield_per_ha = total_production / area if area > 0 else 0
                st.success(t('yield_prediction_success', yield_val=predicted_yield_per_ha))

                st.divider()
                st.subheader(t('market_price_header'))
                try:
                    price_api_key = st.secrets["datagov_api_key"]

                    with st.spinner(t('fetching_prices_spinner')):
                        market_price = get_market_price(crop, price_api_key)
                    
                    if market_price:
                        # Weather-linked price adjustment
                        weather_adjustment = 1.0  # Default multiplier
                        try:
                            # Get current weather for Kerala (assuming general location)
                            weather_api_key = st.secrets["openweathermap_api_key"]
                            weather_data = get_weather("Thiruvananthapuram", weather_api_key)  # Kerala capital as proxy
                            if weather_data and weather_data.get('cod') == 200:
                                current_weather = weather_data['list'][0]
                                temp = current_weather['main']['temp']
                                humidity = current_weather['main']['humidity']
                                
                                # Adjust price based on weather conditions
                                # Higher prices during drought-like conditions (high temp, low humidity)
                                if temp > 35 and humidity < 40:  # Drought conditions
                                    weather_adjustment = 1.15  # 15% increase
                                elif temp > 30 and humidity < 50:
                                    weather_adjustment = 1.08  # 8% increase
                                elif humidity > 80:  # High humidity might affect quality
                                    weather_adjustment = 0.95  # 5% decrease
                                
                                market_price *= weather_adjustment
                                st.info(f"💧 Price adjusted by weather conditions: {weather_adjustment:.0%}")
                        except:
                            pass  # If weather fetch fails, use original price
                        
                        total_value = (total_production / 100) * market_price
                        price_col, value_col = st.columns(2)
                        price_col.metric(label=t('market_price_label'), value=f"₹ {market_price:,.2f}")
                        value_col.metric(label=t('total_value_label'), value=f"₹ {total_value:,.2f}")
                        
                        # Profit Calculation Section
                        st.divider()
                        st.subheader(t('profit_header'))
                        cost_per_ha = st.number_input(t('cost_per_ha_label'), min_value=0.0, value=5000.0, step=100.0, format="%.2f")
                        total_cost = cost_per_ha * area
                        total_profit = total_value - total_cost
                        
                        cost_col, profit_col = st.columns(2)
                        cost_col.metric(label=t('total_cost_label'), value=f"₹ {total_cost:,.2f}")
                        profit_col.metric(label=t('total_profit_label'), value=f"₹ {total_profit:,.2f}")
                        
                    else:
                        st.info(t('price_not_available', crop=translate_if_needed(crop)))

                except (KeyError, FileNotFoundError):
                    st.warning("Market price API key not found in secrets.toml. Please add it to display prices.")
                except Exception:
                    st.error(t('price_api_error'))
            
            except Exception as e:
                st.error(f"An error occurred during yield prediction: {e}")


    # --- MODIFIED: Crop Calendar Page ---
    elif page == "page_crop_calendar":
        st.header(t('crop_calendar_header'))
        kerala_crops_df = load_and_update_data('kerala_crops.csv')
        selected_crop = st.selectbox(t('select_crop_label'), options=kerala_crops_df['Crop'].unique(), format_func=lambda x: translate_if_needed(x))
        
        if selected_crop:
            crop_info = kerala_crops_df[kerala_crops_df['Crop'] == selected_crop].iloc[0]
            st.subheader(t('schedule_for_crop', crop=translate_if_needed(selected_crop)))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=t('sowing_start_label'), value=crop_info['Sowing Start (Kerala)'])
                st.metric(label=t('sowing_end_label'), value=crop_info['Sowing End (Kerala)'])
            with col2:
                st.metric(label=t('harvest_start_label'), value=crop_info['Harvest Start (Approx)'])
                st.metric(label=t('harvest_end_label'), value=crop_info['Harvest End (Approx)'])
            
            st.divider()
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric(label=t('fertiliser_label'), value=crop_info['Fertiliser'])
            with col4:
                duration_value = crop_info['Duration (Days)']
                if "Perennial" not in str(duration_value):
                    duration_value = f"{duration_value} days"
                st.metric(label=t('duration_label'), value=duration_value)

            st.divider()

            # --- SMS Reminder Section with Contact Input ---
            st.subheader("📱 Set Sowing Reminder (SMS)")

            # Get current contact from profile
            user_details = get_user_details(st.session_state['username'])
            saved_contact = user_details.get('contact_number', '') if user_details else ''

            col_contact1, col_contact2 = st.columns([2, 1])
            with col_contact1:
                sms_contact = st.text_input(
                    "📞 Enter your phone number for SMS reminders",
                    value=saved_contact,
                    placeholder="+91XXXXXXXXXX or XXXXXXXXXX",
                    key=f"sms_contact_{selected_crop}",
                    help="Enter 10-digit number or with country code (+91)"
                )

            with col_contact2:
                if saved_contact and not sms_contact:
                    st.info("Using saved number")
                elif sms_contact:
                    st.success("Ready to send SMS")
                else:
                    st.warning("Add phone number")

            # SMS Reminder Buttons
            col_sms1, col_sms2, col_sms3 = st.columns([1.5, 1.5, 1])
            with col_sms1:
                send_reminder = st.button("📤 Send SMS Reminder", key=f"send_sms_{selected_crop}")
            with col_sms2:
                if st.button("ℹ️ View Details", key=f"view_details_{selected_crop}"):
                    details_text = (
                        f"📅 **Sowing Window:** {crop_info['Sowing Start (Kerala)']} to {crop_info['Sowing End (Kerala)']}\n\n"
                        f"🌱 **Crop Duration:** {crop_info['Duration (Days)']} days\n\n"
                        f"🎯 **Harvest Window:** {crop_info['Harvest Start (Approx)']} to {crop_info['Harvest End (Approx)']}\n\n"
                        f"🧪 **Fertiliser:** {crop_info['Fertiliser']}"
                    )
                    current_lang = st.session_state.get('lang', 'en')
                    if current_lang != 'en':
                        details_text = translate_text(details_text, current_lang)
                    st.info(details_text)
            with col_sms3:
                if saved_contact:
                    save_to_profile = st.button("💾 Save to Profile", key=f"save_profile_{selected_crop}")
                else:
                    save_to_profile = False

            # Handle SMS sending
            if send_reminder:
                if not sms_contact:
                    st.error("❌ Please enter a phone number to receive SMS reminders")
                elif len(sms_contact.replace("+", "").replace(" ", "").replace("-", "")) < 10:
                    st.error("❌ Phone number must be at least 10 digits")
                else:
                    with st.spinner("📤 Sending SMS reminder..."):
                        # Send comprehensive SMS with all crop details
                        if send_crop_reminder_sms(sms_contact, selected_crop, crop_info):
                            st.success(f"✅ SMS reminder sent to {sms_contact}!")
                            with st.expander("📋 SMS Details Sent", expanded=True):
                                st.write(f"**Crop:** {selected_crop}")
                                st.write(f"**Sowing:** {crop_info['Sowing Start (Kerala)']} - {crop_info['Sowing End (Kerala)']}")
                                st.write(f"**Duration:** {crop_info['Duration (Days)']} days")
                                st.write(f"**Harvest:** {crop_info['Harvest Start (Approx)']} - {crop_info['Harvest End (Approx)']}")
                                st.write(f"**Fertiliser:** {crop_info['Fertiliser']}")
                        else:
                            st.error("❌ Failed to send SMS. Please check your FAST2SMS API key and try again.")

            # Handle saving to profile
            if save_to_profile and sms_contact:
                if update_user_profile(st.session_state['username'], {'contact_number': sms_contact}):
                    st.success(f"✅ Phone number saved to profile: {sms_contact}")
                    st.session_state['contact_number'] = sms_contact
                    st.rerun()
                else:
                    st.error("❌ Failed to save phone number to profile")
    elif page == "page_weather_report":
        st.header(t('weather_report_header'))
        city = st.text_input(t('enter_city_label'), "Asansol")
        try:
            weather_api_key = st.secrets["openweathermap_api_key"]
        except (KeyError, FileNotFoundError):
            st.error("OpenWeatherMap API key not found. Please add it to your secrets.toml file.")
            st.stop()
        if st.button(t('get_weather_button')):
            if city:
                with st.spinner(f"Fetching weather for {city}..."):
                    weather_data = get_weather(city, weather_api_key)
                if weather_data:
                    cod = weather_data.get("cod")
                    if str(cod) == "200":
                        try:
                            current_condition = weather_data['list'][0]
                            location = f"{weather_data['city']['name']}, {weather_data['city']['country']}"
                            st.subheader(t('current_weather_in', location=location))
                            col1, col2, col3 = st.columns(3)
                            col1.metric(t('temperature_metric'), f"{current_condition['main']['temp']:.1f}°C")
                            col2.metric(t('feels_like_metric'), f"{current_condition['main']['feels_like']:.1f}°C")
                            col3.metric(t('humidity_metric'), f"{current_condition['main']['humidity']}%")
                            st.write(f"**{t('description_label')}** {translate_if_needed(current_condition['weather'][0]['description'].title())}")
                            st.write(f"**{t('wind_label')}** {current_condition['wind']['speed'] * 3.6:.1f} km/h")
                            st.write(f"**{t('precipitation_label')}** {current_condition.get('rain', {}).get('3h', 0)} mm (in last 3h)")
                            st.subheader(t('forecast_header'))
                            forecasts_by_day = {}
                            for forecast in weather_data['list']:
                                date = datetime.datetime.fromtimestamp(forecast['dt']).date()
                                if date not in forecasts_by_day:
                                    forecasts_by_day[date] = []
                                forecasts_by_day[date].append(forecast)
                            for i, (day, day_forecasts) in enumerate(forecasts_by_day.items()):
                                if i > 2:
                                    break
                                min_temp = min(f['main']['temp_min'] for f in day_forecasts)
                                max_temp = max(f['main']['temp_max'] for f in day_forecasts)
                                outlook = next((f['weather'][0]['description'].title() for f in day_forecasts if '12:00:00' in f['dt_txt']), day_forecasts[0]['weather'][0]['description'].title())
                                with st.container(border=True):
                                    st.markdown(f"**{day.strftime('%A, %b %d')}**")
                                    col1, col2, col3 = st.columns(3)
                                    sunrise_ts = datetime.datetime.fromtimestamp(weather_data['city']['sunrise'])
                                    sunset_ts = datetime.datetime.fromtimestamp(weather_data['city']['sunset'])
                                    col1.write(f"**{t('max_temp_label')}** {max_temp:.1f}°C")
                                    col1.write(f"**{t('sunrise_label')}** {sunrise_ts.strftime('%H:%M')}")
                                    col2.write(f"**{t('min_temp_label')}** {min_temp:.1f}°C")
                                    col2.write(f"**{t('sunset_label')}** {sunset_ts.strftime('%H:%M')}")
                                    col3.write(f"**{t('outlook_label')}** {translate_if_needed(outlook)}")
                        except (KeyError, IndexError) as e:
                            st.error(t('error_parsing_weather', e=e))
                    else:
                        message = weather_data.get('message', 'Unknown error.')
                        if str(cod) == "401":
                            st.error(t('weather_api_key_error', message=message))
                        elif str(cod) == "404":
                            st.error(t('weather_city_not_found_error', message=message))
                        else:
                            st.error(t('weather_api_generic_error', cod=cod, message=message))
                else:
                    st.error(t('weather_fetch_error'))
            else:
                st.warning(t('enter_city_warning'))

    elif page == "page_leaf_disease_prediction":
        st.header(t('leaf_disease_header'))
        st.write(t('leaf_disease_intro'))

        if 'last_disease_prediction' not in st.session_state:
            st.session_state['last_disease_prediction'] = None

        if leaf_disease_model is None:
            st.warning(t('model_not_found_warning'))
        else:
            uploaded_file = st.file_uploader(t('uploader_label'), type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption=t('uploaded_image_caption'), width=240)
                with col2:
                    if st.button(t('predict_disease_button')):
                        with st.spinner(t('predicting_spinner')):
                            img_resized = image.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_batch = np.expand_dims(img_array, axis=0)
                        prediction = leaf_disease_model.predict(img_batch)
                        predicted_class_index = np.argmax(prediction[0])
                        CLASS_NAMES = ["Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy", "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy", "Grape Black Rot", "Grape Esca (Black Measles)", "Grape Leaf Blight", "Grape Healthy", "Potato Early Blight", "Potato Late Blight", "Potato Healthy", "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"]
                        st.session_state['last_disease_prediction'] = CLASS_NAMES[predicted_class_index]

            if st.session_state['last_disease_prediction']:
                predicted_disease = st.session_state['last_disease_prediction']
                st.success(t('prediction_result', disease=translate_if_needed(predicted_disease)))
                
                st.divider()
                st.subheader(t('remedy_info_header', disease=translate_if_needed(predicted_disease)))

                disease_key = predicted_disease.lower().replace(" ", "_").replace("(", "").replace(")", "")
                remedy_data = DISEASE_REMEDIES.get(predicted_disease.lower())

                if "healthy" in predicted_disease.lower():
                    st.info(t('healthy_plant_message'))
                elif remedy_data:
                    with st.container(border=True):
                        st.markdown(f"**{t('disease_description_heading')}**")
                        st.write(translate_if_needed(remedy_data['description']))
                        
                        st.markdown(f"**{t('organic_remedies_heading')}**")
                        st.write(translate_if_needed(remedy_data['organic']))

                        st.markdown(f"**{t('chemical_remedies_heading')}**")
                        st.write(translate_if_needed(remedy_data['chemical']))
                else:
                    st.info(f"Remedy information for {predicted_disease} is not yet available.")

    elif page == "page_pest_prediction":
        st.header(t('pest_prediction_header'))

        @st.cache_resource
        def load_pest_model():
            try:
                model_path = "./local_vision_model" 
                processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModelForImageClassification.from_pretrained(model_path)
                return processor, model
            except Exception as e:
                st.error(f"{t('model_loading_error')}\n\n**Details:** Could not load model from local folder. Please ensure 'local_vision_model' folder is in the same directory as app.py and contains the correct files.\n\n`{e}`")
                return None, None

        processor, model = load_pest_model()

        if model is not None and processor is not None:
            uploaded_file_pest = st.file_uploader(t('upload_image_pest'), type=["jpg", "jpeg", "png"])
            
            if uploaded_file_pest is not None:
                image = Image.open(uploaded_file_pest).convert("RGB")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption=t('uploaded_image_caption'), width=240)
                with col2:
                    with st.spinner("Analyzing image..."):
                        inputs = processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_class = model.config.id2label[predicted_class_idx]
                    predicted_class_cleaned = predicted_class.split(',')[0].replace("_", " ").title()
                
                st.success(f"**{t('prediction_pest')}:** {translate_if_needed(predicted_class_cleaned)}")

                # --- NEW: Display Pest Remedies ---
                st.divider()
                st.subheader(t('remedy_info_header_pest', pest=translate_if_needed(predicted_class_cleaned)))

                # Logic to find the correct remedy from the dictionary
                remedy_data = None
                pest_key_cleaned_lower = predicted_class_cleaned.lower()
                for key in PEST_REMEDIES:
                    if key in pest_key_cleaned_lower:
                        remedy_data = PEST_REMEDIES[key]
                        break
                
                if remedy_data:
                    with st.container(border=True):
                        st.markdown(f"**{t('pest_description_heading')}**")
                        st.write(translate_if_needed(remedy_data['description']))
                        
                        st.markdown(f"**{t('organic_remedies_heading')}**")
                        st.write(translate_if_needed(remedy_data['organic']))

                        st.markdown(f"**{t('chemical_remedies_heading')}**")
                        st.write(translate_if_needed(remedy_data['chemical']))
                else:
                    st.info(t('no_remedy_info', pest=translate_if_needed(predicted_class_cleaned)))
                # --- End of New Section ---


    elif page == "page_soil_type_prediction":
        st.header(t('soil_prediction_header'))
        st.write(t('soil_prediction_intro'))
        if 'last_soil_prediction' not in st.session_state:
            st.session_state['last_soil_prediction'] = None
        if soil_model is None:
            st.warning(t('soil_model_not_found_warning'))
        else:
            uploaded_file = st.file_uploader(t('upload_soil_image_label'), type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption=t('uploaded_image_caption'), width=220)
                with col2:
                    if st.button(t('predict_soil_button')):
                        with st.spinner(t('analyzing_soil_spinner')):
                            img_resized = image.resize((224, 224))
                            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                            img_batch = np.expand_dims(img_array, axis=0)
                            img_preprocessed = preprocess_input(img_batch)
                            predictions = soil_model.predict(img_preprocessed)
                            decoded = decode_predictions(predictions, top=1)[0]
                            predicted_label = decoded[0][1]
                            st.write(f"General Prediction by Model: `{predicted_label}`")
                            prediction_map = {
                                "Alluvial Soil": ["sandbar", "valley", "river", "loam", "seashore"],
                                "Red Soil": ["desert", "volcano", "badlands", "rock", "mountain"],
                                "Black Soil": ["volcano", "lava", "promontory"],
                                "Clay Soil": ["earthstar", "mud", "wetland", "plow"],
                                "Laterite Soil": ["cliff", "promontory", "rock_wall"]
                            }
                            final_prediction = "Could not determine soil type"
                            for soil_type, keywords in prediction_map.items():
                                if any(keyword in predicted_label.lower() for keyword in keywords):
                                    final_prediction = soil_type
                                    break
                            st.session_state['last_soil_prediction'] = final_prediction
        if st.session_state['last_soil_prediction']:
            predicted_soil = st.session_state['last_soil_prediction']
            if predicted_soil != "Could not determine soil type":
                st.success(t('soil_prediction_result', soil_type=translate_if_needed(predicted_soil)))
                st.divider()
                st.subheader(t('soil_info_header', soil_type=translate_if_needed(predicted_soil)))
                soil_key = predicted_soil.lower()
                soil_data = SOIL_INFO.get(soil_key)
                if soil_data:
                    with st.container(border=True):
                        st.markdown(f"**{t('description_heading')}**")
                        st.write(translate_if_needed(soil_data['description']))
                        st.markdown(f"**{t('soil_characteristics_heading')}**")
                        st.write(translate_if_needed(soil_data['characteristics']))
                        st.markdown(f"**{t('soil_suitable_crops_heading')}**")
                        st.write(translate_if_needed(soil_data['suitable_crops']))
                else:
                    st.info(f"Detailed information for {predicted_soil} is not yet available.")
            else:
                st.warning("The model could not confidently identify a specific soil type from this image. Please try another photo with a clearer view of the soil.")

    elif page == "page_translator":
        st.header(t('translator_header'))
        text = st.text_area(t('translator_intro'))
        if st.button(t('translate_button')):
            if text.strip():
                try:
                    st.success(f"🇮🇳 Telugu: {GoogleTranslator(source='auto', target='te').translate(text)}")
                    st.success(f"🇮🇳 Tamil: {GoogleTranslator(source='auto', target='ta').translate(text)}")
                    st.success(f"🇮🇳 Malayalam: {GoogleTranslator(source='auto', target='ml').translate(text)}")
                    st.success(f"🇮🇳 Hindi: {GoogleTranslator(source='auto', target='hi').translate(text)}")
                    st.success(f"🇮🇳 Bengali: {GoogleTranslator(source='auto', target='bn').translate(text)}")
                    st.success(f"🇮🇳 Marathi: {GoogleTranslator(source='auto', target='mr').translate(text)}")
                    st.success(f"🇮🇳 Punjabi: {GoogleTranslator(source='auto', target='pa').translate(text)}")
                except Exception as e:
                    st.error(t('translation_failed_error', e=e))
            else:
                st.warning(t('enter_text_warning'))

    # --- NEW COMMUNITY FORUM PAGE ---
    elif page == "page_community_forum":
        st.header(t('forum_header'))
        st.write(t('forum_intro'))

        with st.expander(t('create_post_header')):
            with st.form("new_post_form", clear_on_submit=True):
                post_title = st.text_input(t('post_title_label'), placeholder=t('post_title_placeholder'))
                post_content = st.text_area(t('post_content_label'), placeholder=t('post_content_placeholder'), height=200)
                post_image = st.file_uploader(t('upload_image_label'), type=['png', 'jpg', 'jpeg'])
                
                submitted = st.form_submit_button(t('submit_post_button'))
                if submitted:
                    if post_title and post_content:
                        image_b64 = None
                        if post_image:
                            bytes_data = post_image.getvalue()
                            image_b64 = base64.b64encode(bytes_data).decode()
                        
                        add_forum_post(
                            username=st.session_state['username'],
                            full_name=st.session_state['full_name'],
                            title=post_title,
                            content=post_content,
                            image_b64=image_b64
                        )
                        st.success(t('post_success_message'))
                        st.rerun()
                    else:
                        st.warning("Title and content are required.")

        st.divider()
        st.subheader(t('all_posts_header'))
        
        all_posts = get_all_forum_posts()

        if not all_posts:
            st.info(t('no_posts_message'))
        else:
            for post in all_posts:
                post_id = str(post['_id'])
                with st.container(border=True):
                    # Format timestamp
                    utc = timezone('UTC')
                    ist = timezone('Asia/Kolkata')
                    utc_time = post['timestamp'].replace(tzinfo=utc)
                    ist_time = utc_time.astimezone(ist)
                    date_str = ist_time.strftime("%d %b %Y, %I:%M %p")

                    st.markdown(f"#### {post['title']}")
                    st.caption(t('posted_by', username=post['full_name'], date=date_str))
                    st.markdown(post['content'])

                    if post.get('image_b64'):
                        try:
                            img_data = base64.b64decode(post['image_b64'])
                            st.image(img_data)
                        except Exception:
                            st.warning("Could not display the attached image.")

                    # --- Interaction Buttons (Like & Reply) ---
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        like_count = len(post.get('likes', []))
                        if st.button(t('like_button', count=like_count), key=f"like_{post_id}"):
                            toggle_like_post(post_id, st.session_state['username'])
                            st.rerun()
                    
                    with col2:
                        replies = get_replies_for_post(post_id)
                        reply_count = len(replies)
                        
                        # Toggle to show/hide replies
                        toggle_label = t('view_replies', count=reply_count) if reply_count > 0 else t('reply_button')
                        show_replies = st.checkbox(toggle_label, key=f"toggle_{post_id}")

                    if show_replies:
                        # Display existing replies
                        if replies:
                            st.markdown(f"**{t('replies_header')}:**")
                            for reply in replies:
                                with st.container(border=True):
                                    reply_utc_time = reply['timestamp'].replace(tzinfo=utc)
                                    reply_ist_time = reply_utc_time.astimezone(ist)
                                    reply_date_str = reply_ist_time.strftime("%d %b %Y, %I:%M %p")
                                    st.caption(t('posted_by', username=reply['full_name'], date=reply_date_str))
                                    st.write(reply['content'])
                        
                        # Form to add a new reply
                        with st.form(key=f"reply_form_{post_id}", clear_on_submit=True):
                            reply_content = st.text_input(t('add_reply_placeholder'), key=f"reply_input_{post_id}", label_visibility="collapsed")
                            reply_submitted = st.form_submit_button(t('submit_reply_button'))
                            if reply_submitted and reply_content:
                                add_forum_reply(
                                    post_id=post_id,
                                    username=st.session_state['username'],
                                    full_name=st.session_state['full_name'],
                                    content=reply_content
                                )
                                st.rerun()

