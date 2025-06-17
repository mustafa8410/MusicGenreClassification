package com.example.MusicGenreClassification.view;

import com.example.MusicGenreClassification.entity.Prediction;
import com.example.MusicGenreClassification.entity.User;
import com.example.MusicGenreClassification.repository.PredictionRepository;
import com.example.MusicGenreClassification.repository.UserRepository;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.security.Principal;
import java.util.List;

@Controller
public class AudioPageController {

    private final UserRepository userRepository;
    private final PredictionRepository predictionRepository;

    public AudioPageController(UserRepository userRepository, PredictionRepository predictionRepository) {
        this.userRepository = userRepository;
        this.predictionRepository = predictionRepository;
    }

    @GetMapping("/audio/upload")
    public String showUploadPage() {
        return "audio_upload";
    }

    @GetMapping("/audio/history")
    public String showUserHistory(Model model, Principal principal) {
        if (principal == null) {
            // Not logged in, redirect to login
            return "redirect:/login";
        }
        // Fetch user's predictions (with audio) from DB
        String username = principal.getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found!"));
        List<Prediction> predictions = predictionRepository
                .findByAudio_UserIdOrderByPredictionDateDesc(user.getId());
        model.addAttribute("predictions", predictions);
        return "history";
    }
}
