package com.example.MusicGenreClassification.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class HistoryResponse {
    private Long predictionId;
    private String genre;
    private double confidence;
    private LocalDateTime predictionDate;

    private String audioUrl;
    private String displayName;
    private LocalDateTime uploadDate;
}