package com.example.MusicGenreClassification.service;

import com.example.MusicGenreClassification.entity.Audio;
import com.example.MusicGenreClassification.entity.Prediction;
import com.example.MusicGenreClassification.entity.User;
import com.example.MusicGenreClassification.repository.AudioRepository;
import com.example.MusicGenreClassification.repository.PredictionRepository;
import com.example.MusicGenreClassification.repository.UserRepository;
import com.example.MusicGenreClassification.response.HistoryResponse;
import com.example.MusicGenreClassification.response.PredictionResponse;
import jakarta.servlet.http.HttpSession;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.Principal;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@Slf4j
public class AudioService {

    private final AudioRepository audioRepository;
    private final UserRepository userRepository;
    private final PredictionRepository predictionRepository;

    private final RestTemplate restTemplate = new RestTemplate();

    public AudioService(AudioRepository audioRepository, UserRepository userRepository, PredictionRepository predictionRepository) {
        this.audioRepository = audioRepository;
        this.userRepository = userRepository;
        this.predictionRepository = predictionRepository;
    }

    private final String uploadDirectory = "uploads";
    public PredictionResponse uploadAudio(MultipartFile file, HttpSession session, Principal principal) {
        try {
            // saving the audio file
            String fileName = UUID.randomUUID() + "_" + file.getOriginalFilename();
            Path uploadPath = Paths.get(uploadDirectory);
            if(!Files.exists(uploadPath))
                Files.createDirectories(uploadPath);
            Path filePath = uploadPath.resolve(fileName);
            Files.copy(file.getInputStream(), filePath);
            log.info("Audio file uploaded successfully to path: {}", filePath.toString());

            // saving the audio metadata to the database
            Audio audio = new Audio();
            audio.setFileName(fileName);
            audio.setUploadDate(LocalDateTime.now());
            // Check for logged-in user
            if (principal != null) {
                User user = userRepository.findByUsername(principal.getName())
                        .orElseThrow(() -> new RuntimeException("User not found!"));
                audio.setUser(user);
            } else {
                // Guest: set guestId from session
                String guestId = (String) session.getAttribute("guestId");
                audio.setGuestId(guestId);
            }
            audio = audioRepository.save(audio);
            log.info("Audio metadata saved to database with ID: {}", audio.getId());

            // calling the ml microservice
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(filePath.toFile()));
            log.info("Preparing to send the audio file to ML microservice for prediction");
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<PredictionResponse> response = restTemplate
                    .postForEntity("http://localhost:8000/predict", requestEntity, PredictionResponse.class);
            PredictionResponse predictionResponse = response.getBody();
            log.info("Received prediction response from ML microservice: {}", predictionResponse.toString());

            // saving the prediction response to the database
            Prediction prediction = new Prediction();
            prediction.setAudio(audio);
            prediction.setGenre(predictionResponse.getGenre());
            prediction.setConfidence(predictionResponse.getConfidence());
            prediction.setPredictionDate(LocalDateTime.now());
            prediction = predictionRepository.save(prediction);
            log.info("Prediction saved to database with ID: {}", prediction.getId());

            // returning the prediction response
            return predictionResponse;

        }
        catch (Exception e) {
            throw new RuntimeException(e.getMessage(), e);
        }
    }

    public ResponseEntity<Resource> downloadAudio(Long audioId) {
        Audio audio = audioRepository.findById(audioId)
                .orElseThrow(() -> new RuntimeException("Audio not found with ID: " + audioId));
        Path filePath = Paths.get(uploadDirectory, audio.getFileName());
        Resource fileResource = new FileSystemResource(filePath.toFile());
        if (!fileResource.exists()) {
            throw new RuntimeException("File not found: " + filePath.toString());
        }
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + audio.getFileName() + "\"")
                .contentType(MediaType.parseMediaType("audio/wav"))
                .body(fileResource);
    }

//    public List<HistoryResponse> getUserAudios(Long userId) {
//        User user = userRepository.findById(userId)
//                .orElseThrow(() -> new RuntimeException("User not found with ID: " + userId));
//        List<Prediction> predictions = predictionRepository.findByAudio_UserIdOrderByPredictionDateDesc(userId);
//        return predictions.stream().map( prediction -> {
//            Audio audio = prediction.getAudio();
//            HistoryResponse historyResponse = new HistoryResponse();
//            historyResponse.setPredictionId(prediction.getId());
//            historyResponse.setGenre(prediction.getGenre());
//            historyResponse.setConfidence(prediction.getConfidence());
//            historyResponse.setPredictionDate(prediction.getPredictionDate());
//            historyResponse.setAudioUrl("/api/audio/download/" + audio.getId());
//            historyResponse.setDisplayName("recording-" + audio.getUploadDate().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")) + ".wav");
//            historyResponse.setUploadDate( audio.getUploadDate() );
//            return historyResponse;
//        } ).collect(Collectors.toList());
//
//    }
}
