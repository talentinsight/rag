#!/usr/bin/env python3
"""
Setup script for Advanced PII Detection
Installs and configures all necessary components
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and handle errors"""
    logger.info(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Failed: {e.stderr}")
        return False

def main():
    """Setup advanced PII detection components"""
    logger.info("🚀 Setting up Advanced PII Detection System")
    logger.info("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        sys.exit(1)
    
    logger.info(f"✅ Python version: {sys.version}")
    
    # Install core requirements
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install presidio-analyzer presidio-anonymizer", "Installing Microsoft Presidio"),
        ("pip install spacy>=3.6.0", "Installing spaCy"),
        ("pip install torch transformers", "Installing PyTorch and Transformers"),
        ("python -m spacy download en_core_web_sm", "Downloading spaCy English model"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    logger.info(f"\n📊 Installation Summary:")
    logger.info(f"   Successful: {success_count}/{len(commands)}")
    
    if success_count == len(commands):
        logger.info("🎉 Advanced PII Detection setup completed successfully!")
        logger.info("\n📋 Next Steps:")
        logger.info("   1. Restart your application")
        logger.info("   2. The system will automatically use advanced detection")
        logger.info("   3. Check logs for 'Advanced PII Detector initialized' message")
        
        # Test the installation
        logger.info("\n🧪 Testing installation...")
        test_advanced_pii()
    else:
        logger.warning("⚠️  Some components failed to install")
        logger.info("   The system will fall back to regex-based detection")

def test_advanced_pii():
    """Test the advanced PII detection installation"""
    try:
        # Test Presidio
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
        
        test_text = "My name is John Doe and my email is john@example.com"
        results = analyzer.analyze(text=test_text, language='en')
        
        logger.info(f"✅ Presidio test successful - Found {len(results)} entities")
        
        # Test transformers
        from transformers import pipeline
        logger.info("✅ Transformers test successful")
        
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy test successful")
        
        logger.info("🎯 All components working correctly!")
        
    except Exception as e:
        logger.warning(f"⚠️  Test failed: {e}")
        logger.info("   System will use fallback detection methods")

if __name__ == "__main__":
    main()
