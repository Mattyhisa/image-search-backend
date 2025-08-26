from flask import Flask, request, jsonify
import requests
import torch
import clip
from PIL import Image
import io
import faiss
import numpy as np
from flask_cors import CORS
import os

# Initialiser Flask
app = Flask(__name__)
CORS(app)  # Permettre les requêtes depuis https://rmatthieu.art

# Token Mapillary (utiliser variable d'environnement sur Render)
MAPILLARY_TOKEN = os.getenv('MAPILLARY_TOKEN', 'MLY|30313244324986047|4b7ed9124be521803815eea0ecb98298')

# Charger CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Calculer la bounding box pour Mapillary
def calculate_bbox(lat, lon, radius):
    lat = float(lat)
    lon = float(lon)
    radius = float(radius)
    lat_rad = lat * np.pi / 180
    delta_lat = radius / 111000
    delta_lon = radius / (111000 * np.cos(lat_rad))
    return {
        'west': lon - delta_lon,
        'south': lat - delta_lat,
        'east': lon + delta_lon,
        'north': lat + delta_lat,
    }

# Télécharger une image depuis une URL
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Erreur téléchargement image {url}: {e}")
        return None

# Route pour la recherche d'images similaires
@app.route('/search-similar', methods=['POST'])
def search_similar():
    try:
        # Vérifier les paramètres
        if 'image' not in request.files or 'lat' not in request.form or 'lon' not in request.form or 'radius' not in request.form:
            return jsonify({'error': 'Image, latitude, longitude ou rayon manquant.'}), 400

        image_file = request.files['image']
        lat = request.form['lat']
        lon = request.form['lon']
        radius = request.form['radius']
        num_results = int(request.form.get('num_results', 10))
        threshold = float(request.form.get('threshold', 0.8))

        # Valider les inputs
        if image_file.mimetype not in ['image/jpeg', 'image/png']:
            return jsonify({'error': 'Image doit être JPG ou PNG.'}), 400
        if num_results < 1 or num_results > 50:
            return jsonify({'error': 'Nombre de résultats doit être entre 1 et 50.'}), 400
        if threshold < 0 or threshold > 1:
            return jsonify({'error': 'Seuil doit être entre 0 et 1.'}), 400

        # Calculer bbox
        bbox = calculate_bbox(lat, lon, radius)
        bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

        # Appeler Mapillary
        mapillary_url = f"https://graph.mapillary.com/images?access_token={MAPILLARY_TOKEN}&fields=id,thumb_original_url,geometry,captured_at&bbox={bbox_str}"
        response = requests.get(mapillary_url, headers={'User-Agent': 'MapillaryImageSearch/1.0'}, timeout=10)
        data = response.json()
        if 'data' not in data or not data['data']:
            return jsonify({'error': 'Aucune image trouvée sur Mapillary.'}), 404

        candidates = data['data'][:1000]  # Limiter à 1000 pour performance

        # Préparer l'image uploadée
        uploaded_img = Image.open(image_file)
        uploaded_preprocessed = preprocess(uploaded_img).unsqueeze(0).to(device)
        with torch.no_grad():
            uploaded_embedding = model.encode_image(uploaded_preprocessed).cpu().numpy()

        # Prompt pour filtrer les maisons (inclut images partielles)
        house_text = clip.tokenize(["a photo of a house facade"]).to(device)
        with torch.no_grad():
            house_embedding = model.encode_text(house_text).cpu().numpy()

        # Extraire embeddings des candidates
        embeddings = []
        filtered_candidates = []
        for cand in candidates:
            img = download_image(cand['thumb_original_url'])
            if img is None:
                continue
            img_preprocessed = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_embedding = model.encode_image(img_preprocessed).cpu().numpy()
                # Filtre maison: score > 0.5 (accepte images partielles)
                house_sim = np.dot(img_embedding, house_embedding.T) / (np.linalg.norm(img_embedding) * np.linalg.norm(house_embedding))
                if house_sim > 0.5:
                    embeddings.append(img_embedding.flatten())
                    filtered_candidates.append(cand)

        if not embeddings:
            return jsonify({'error': 'Aucune image de maison trouvée après filtrage.'}), 404

        # Index FAISS pour recherche rapide
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Recherche des similaires
        faiss.normalize_L2(uploaded_embedding)
        distances, indices = index.search(uploaded_embedding, len(embeddings))

        # Collecter résultats
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= threshold:
                cand = filtered_candidates[idx]
                results.append({
                    'id': cand['id'],
                    'url': cand['thumb_original_url'],
                    'lat': cand['geometry']['coordinates'][1],
                    'lon': cand['geometry']['coordinates'][0],
                    'score': float(dist)
                })
            if len(results) >= num_results:
                break

        # Trier par score descendant
        results.sort(key=lambda x: x['score'], reverse=True)

        # Pour plus tard : Reverse geocoding avec Nominatim
        # from geopy.geocoders import Nominatim
        # geolocator = Nominatim(user_agent="image-search")
        # for res in results:
        #     location = geolocator.reverse((res['lat'], res['lon']))
        #     res['address'] = location.address if location else "Adresse inconnue"

        return jsonify({'images': results})

    except Exception as e:
        print(f"Erreur serveur: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render utilise PORT
    app.run(host='0.0.0.0', port=port, debug=False)
