import os
import requests
import ftplib
import pandas as pd
import numpy as np
import netCDF4 as nc
import psycopg2
from psycopg2.extras import Json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import xarray as xr
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSON, ARRAY
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()

class ArgoProfile(Base):
    """ARGO profile data model"""
    __tablename__ = 'argo_profiles'
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False)
    wmo_number = Column(String(20), index=True)
    
    # Location and time
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    juld = Column(DateTime, nullable=False, index=True)  # Julian day
    
    # Profile metadata
    ocean_region = Column(String(50), index=True)
    data_centre = Column(String(10))
    direction = Column(String(1))  # A=ascending, D=descending
    data_mode = Column(String(1))  # R=real-time, D=delayed-mode, A=adjusted
    
    # Profile data (stored as JSON arrays)
    pressure = Column(JSON)  # dbar
    temperature = Column(JSON)  # degrees Celsius
    salinity = Column(JSON)  # psu
    
    # BGC parameters (optional)
    dissolved_oxygen = Column(JSON)  # umol/kg
    nitrate = Column(JSON)  # umol/kg
    ph_in_situ = Column(JSON)
    chlorophyll = Column(JSON)  # mg/m3
    backscattering = Column(JSON)
    fluorescence_cdom = Column(JSON)
    
    # Quality control flags
    position_qc = Column(String(1))
    time_qc = Column(String(1))
    profile_pres_qc = Column(ARRAY(String))
    profile_temp_qc = Column(ARRAY(String))
    profile_psal_qc = Column(ARRAY(String))
    
    # File metadata
    source_file = Column(String(255))
    data_type = Column(String(10))  # core, bgc, etc.
    processed_at = Column(DateTime, default=datetime.utcnow)

class ArgoFloat(Base):
    """ARGO float metadata model"""
    __tablename__ = 'argo_floats'
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), unique=True, nullable=False, index=True)
    wmo_number = Column(String(20), unique=True, index=True)
    
    # Float characteristics
    project_name = Column(String(100))
    pi_name = Column(String(100))
    platform_type = Column(String(50))
    float_serial_no = Column(String(50))
    firmware_version = Column(String(20))
    
    # Deployment info
    deployment_date = Column(DateTime)
    deployment_latitude = Column(Float)
    deployment_longitude = Column(Float)
    deployment_platform = Column(String(100))
    
    # Current status
    last_location_date = Column(DateTime)
    last_latitude = Column(Float)
    last_longitude = Column(Float)
    status = Column(String(20))  # active, inactive, etc.
    
    # Sensor information
    sensors = Column(JSON)  # List of sensors
    data_centre = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class DataFetcher:
    """Handles fetching ARGO data from various sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gdac_urls = [
            "https://data-argo.ifremer.fr",
            "https://usgodae.org/ftp/outgoing/argo"
        ]
        self.incois_ftp = "ftp.incois.gov.in"
        
    def fetch_incois_realtime_data(self, date_range: Optional[tuple] = None) -> List[str]:
        """
        Fetch real-time ARGO data from INCOIS FTP server
        Returns list of downloaded NetCDF file paths
        """
        downloaded_files = []
        
        try:
            # Connect to INCOIS FTP server
            ftp = ftplib.FTP(self.incois_ftp)
            ftp.login()  # Anonymous login
            
            # Navigate to ARGO data directory
            ftp.cwd('/pub/argo')
            
            # Get file listing
            files = []
            ftp.retrlines('LIST', files.append)
            
            # Filter for NetCDF files from specified date range
            today = datetime.now()
            if date_range is None:
                # Default to last 7 days
                start_date = today - timedelta(days=7)
                end_date = today
            else:
                start_date, end_date = date_range
            
            netcdf_files = [f for f in files if f.endswith('.nc')]
            
            # Download files
            download_dir = self.config.get('download_dir', './data/raw/')
            os.makedirs(download_dir, exist_ok=True)
            
            for file_info in netcdf_files:
                filename = file_info.split()[-1]
                local_path = os.path.join(download_dir, filename)
                
                # Check if file already exists
                if not os.path.exists(local_path):
                    try:
                        with open(local_path, 'wb') as local_file:
                            ftp.retrbinary(f'RETR {filename}', local_file.write)
                        downloaded_files.append(local_path)
                        logger.info(f"Downloaded: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to download {filename}: {e}")
            
            ftp.quit()
            
        except Exception as e:
            logger.error(f"Failed to connect to INCOIS FTP: {e}")
        
        return downloaded_files
    
    def fetch_gdac_data(self, platform_numbers: List[str] = None) -> List[str]:
        """
        Fetch ARGO data from Global Data Assembly Centers
        """
        downloaded_files = []
        
        for gdac_url in self.gdac_urls:
            try:
                if platform_numbers:
                    for platform_number in platform_numbers:
                        # Construct URL for specific float
                        float_url = f"{gdac_url}/dac/incois/{platform_number}/profiles/"
                        # Implementation for downloading specific float data
                        pass
                else:
                    # Fetch recent data
                    # Implementation for downloading recent data
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to fetch from {gdac_url}: {e}")
                continue
        
        return downloaded_files

class NetCDFProcessor:
    """Processes NetCDF files and extracts ARGO data"""
    
    def __init__(self):
        self.supported_variables = {
            # Core variables
            'TEMP': 'temperature',
            'PSAL': 'salinity', 
            'PRES': 'pressure',
            
            # BGC variables
            'DOXY': 'dissolved_oxygen',
            'NITRATE': 'nitrate',
            'PH_IN_SITU_TOTAL': 'ph_in_situ',
            'CHLA': 'chlorophyll',
            'BBP700': 'backscattering',
            'CDOM': 'fluorescence_cdom'
        }
    
    def process_netcdf_file(self, file_path: str) -> Dict:
        """
        Process a single NetCDF file and extract ARGO profile data
        """
        try:
            # Open NetCDF file using xarray for better handling
            ds = xr.open_dataset(file_path)
            
            profile_data = {
                'source_file': os.path.basename(file_path),
                'profiles': []
            }
            
            # Extract dimensions
            n_prof = ds.dims.get('N_PROF', 1)
            n_levels = ds.dims.get('N_LEVELS', 0)
            
            for prof_idx in range(n_prof):
                profile = self._extract_profile(ds, prof_idx)
                if profile:
                    profile_data['profiles'].append(profile)
            
            ds.close()
            return profile_data
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_profile(self, ds: xr.Dataset, prof_idx: int) -> Optional[Dict]:
        """Extract individual profile data from NetCDF dataset"""
        try:
            profile = {}
            
            # Basic metadata
            profile['platform_number'] = str(ds['PLATFORM_NUMBER'].values[prof_idx]).strip()
            profile['cycle_number'] = int(ds['CYCLE_NUMBER'].values[prof_idx])
            
            # Location and time
            profile['latitude'] = float(ds['LATITUDE'].values[prof_idx])
            profile['longitude'] = float(ds['LONGITUDE'].values[prof_idx])
            
            # Convert Julian day to datetime
            juld = ds['JULD'].values[prof_idx]
            if not np.isnan(juld):
                # ARGO uses days since 1950-01-01
                reference_date = datetime(1950, 1, 1)
                profile['juld'] = reference_date + timedelta(days=float(juld))
            
            # Profile direction and data mode
            profile['direction'] = str(ds['DIRECTION'].values[prof_idx]).strip()
            profile['data_mode'] = str(ds['DATA_MODE'].values[prof_idx]).strip()
            
            # Extract profile data
            for nc_var, db_field in self.supported_variables.items():
                if nc_var in ds.variables:
                    data = ds[nc_var].values[prof_idx]
                    # Remove fill values and convert to list
                    valid_data = data[~np.isnan(data)].tolist()
                    profile[db_field] = valid_data if valid_data else None
                    
                    # Extract QC flags if available
                    qc_var = f"{nc_var}_QC"
                    if qc_var in ds.variables:
                        qc_data = ds[qc_var].values[prof_idx]
                        profile[f"{db_field}_qc"] = qc_data.tolist()
            
            # Extract additional metadata
            if 'DATA_CENTRE' in ds.variables:
                profile['data_centre'] = str(ds['DATA_CENTRE'].values[prof_idx]).strip()
            
            # Determine ocean region based on coordinates
            profile['ocean_region'] = self._determine_ocean_region(
                profile['latitude'], profile['longitude']
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error extracting profile {prof_idx}: {e}")
            return None
    
    def _determine_ocean_region(self, lat: float, lon: float) -> str:
        """Determine ocean region based on coordinates"""
        # Simplified ocean boundary determination
        if 30 <= lon <= 120 and -60 <= lat <= 30:
            return "Indian Ocean"
        elif -80 <= lon <= 30 and -60 <= lat <= 70:
            return "Atlantic Ocean"  
        elif 120 <= lon <= 180 or -180 <= lon <= -80:
            return "Pacific Ocean"
        elif lat < -60:
            return "Southern Ocean"
        elif lat > 60:
            return "Arctic Ocean"
        else:
            return "Unknown"

class DatabaseManager:
    """Manages PostgreSQL database operations"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def insert_profile_data(self, profile_data: Dict) -> bool:
        """Insert processed profile data into database"""
        try:
            for profile in profile_data.get('profiles', []):
                # Check if profile already exists
                existing = self.session.query(ArgoProfile).filter_by(
                    platform_number=profile['platform_number'],
                    cycle_number=profile['cycle_number']
                ).first()
                
                if existing:
                    logger.info(f"Profile already exists: {profile['platform_number']}-{profile['cycle_number']}")
                    continue
                
                # Create new profile record
                argo_profile = ArgoProfile(
                    platform_number=profile['platform_number'],
                    cycle_number=profile['cycle_number'],
                    latitude=profile['latitude'],
                    longitude=profile['longitude'],
                    juld=profile.get('juld'),
                    ocean_region=profile.get('ocean_region'),
                    data_centre=profile.get('data_centre'),
                    direction=profile.get('direction'),
                    data_mode=profile.get('data_mode'),
                    pressure=profile.get('pressure'),
                    temperature=profile.get('temperature'),
                    salinity=profile.get('salinity'),
                    dissolved_oxygen=profile.get('dissolved_oxygen'),
                    nitrate=profile.get('nitrate'),
                    ph_in_situ=profile.get('ph_in_situ'),
                    chlorophyll=profile.get('chlorophyll'),
                    backscattering=profile.get('backscattering'),
                    fluorescence_cdom=profile.get('fluorescence_cdom'),
                    source_file=profile_data['source_file'],
                    data_type='core'  # Can be enhanced to detect BGC data
                )
                
                self.session.add(argo_profile)
            
            self.session.commit()
            logger.info(f"Successfully inserted {len(profile_data.get('profiles', []))} profiles")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Database insertion failed: {e}")
            return False
    
    def get_profiles_by_region(self, lat_range: tuple, lon_range: tuple, 
                              date_range: tuple = None) -> List[Dict]:
        """Retrieve profiles by geographic and temporal criteria"""
        query = self.session.query(ArgoProfile).filter(
            ArgoProfile.latitude.between(lat_range[0], lat_range[1]),
            ArgoProfile.longitude.between(lon_range[0], lon_range[1])
        )
        
        if date_range:
            query = query.filter(
                ArgoProfile.juld.between(date_range[0], date_range[1])
            )
        
        results = query.all()
        return [self._profile_to_dict(profile) for profile in results]
    
    def _profile_to_dict(self, profile: ArgoProfile) -> Dict:
        """Convert SQLAlchemy model to dictionary"""
        return {
            'id': profile.id,
            'platform_number': profile.platform_number,
            'cycle_number': profile.cycle_number,
            'latitude': profile.latitude,
            'longitude': profile.longitude,
            'juld': profile.juld.isoformat() if profile.juld else None,
            'ocean_region': profile.ocean_region,
            'temperature': profile.temperature,
            'salinity': profile.salinity,
            'pressure': profile.pressure,
            'dissolved_oxygen': profile.dissolved_oxygen,
            'nitrate': profile.nitrate,
            'ph_in_situ': profile.ph_in_situ,
            'chlorophyll': profile.chlorophyll
        }

class ArgoDataPipeline:
    """Main pipeline for fetching, processing, and storing ARGO data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fetcher = DataFetcher(config)
        self.processor = NetCDFProcessor()
        self.db_manager = DatabaseManager(config['database_url'])
    
    def run_pipeline(self, source: str = 'incois', date_range: tuple = None):
        """Run the complete data pipeline"""
        logger.info("Starting ARGO data pipeline...")
        
        # Step 1: Fetch data
        if source == 'incois':
            downloaded_files = self.fetcher.fetch_incois_realtime_data(date_range)
        elif source == 'gdac':
            downloaded_files = self.fetcher.fetch_gdac_data()
        else:
            logger.error(f"Unsupported data source: {source}")
            return
        
        logger.info(f"Downloaded {len(downloaded_files)} files")
        
        # Step 2: Process NetCDF files
        total_profiles = 0
        for file_path in downloaded_files:
            logger.info(f"Processing: {file_path}")
            
            profile_data = self.processor.process_netcdf_file(file_path)
            if profile_data:
                # Step 3: Store in database
                success = self.db_manager.insert_profile_data(profile_data)
                if success:
                    total_profiles += len(profile_data.get('profiles', []))
        
        logger.info(f"Pipeline completed. Processed {total_profiles} total profiles")

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        'database_url': 'postgresql://postgres:142003@localhost:5432/argo_db',
        'download_dir': './data/raw/',
        'log_level': 'INFO'
    }
    
    # Initialize and run pipeline
    pipeline = ArgoDataPipeline(config)
    
    # Run for last 7 days of INCOIS data
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    pipeline.run_pipeline(source='incois', date_range=(start_date, end_date))