package com.example.MusicGenreClassification.controller;

import com.example.MusicGenreClassification.entity.Audio;
import com.example.MusicGenreClassification.response.HistoryResponse;
import com.example.MusicGenreClassification.response.PredictionResponse;
import com.example.MusicGenreClassification.service.AudioService;
import jakarta.servlet.http.HttpSession;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.security.Principal;
import java.util.List;

@RestController
@RequestMapping("/audio")
@Slf4j
public class AudioController {
    private final AudioService audioService;
    public AudioController(AudioService audioService) {
        this.audioService = audioService;
    }

    @PostMapping("/upload")
    public ResponseEntity<PredictionResponse> uploadAudio(
            @RequestParam("file") MultipartFile file, HttpSession session, Principal principal) {
        log.info("Received file upload request for file: {}", file.getOriginalFilename());
        PredictionResponse response = audioService.uploadAudio(file, session, principal);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/download/{audioId}")
    public ResponseEntity<Resource> downloadAudio(@PathVariable Long audioId) {
        ResponseEntity<Resource> resourceResponseEntity = audioService.downloadAudio(audioId);
        return resourceResponseEntity;
    }

//    @GetMapping("/history")
//    @ResponseStatus(HttpStatus.OK)
//    public List<HistoryResponse> getUserAudios(@RequestParam Long userId) {
//       return audioService.getUserAudios(userId);
//    }

}
